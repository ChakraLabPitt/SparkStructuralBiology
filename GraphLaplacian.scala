import org.apache.spark._
import org.apache.spark.SparkConf
import scala.math
import org.apache.spark.mllib.linalg.distributed.{RowMatrix,IndexedRowMatrix, MatrixEntry, CoordinateMatrix}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import java.io.FileWriter
import scala.util.Marshal
import scala.io.Source
import scala.collection.immutable
import java.io._

object KirchhoffApp {
def main(args: Array[String]) {
	val conf = new SparkConf().setAppName("KirchhoffApp")
	val sc = new SparkContext(conf) 

	class ParsePDB(val PDBname: String){
		/** PARSE PDB Class */
		var textFile = sc.textFile(PDBname);
		var pairs=textFile.map(line => (line.split("\\s+")(0),line));
		var ATOMlines = pairs.filter{case (key, value) => value.split("\\s+")(0)=="ATOM"};
		def chain_coor = ATOMlines.map{case (key, value) => (value.toString.substring(21,22),value.toString.substring(30,38).toFloat, value.toString.substring(39,46).toFloat,value.toString.substring(47,54).toFloat )};
		def Chains_Natoms=ATOMlines.map{case (key, value) => (value.toString.substring(21,22),1)}.reduceByKey(_ + _).sortByKey();
		def Chains = Chains_Natoms.map(line => line._1);
		def Natoms = Chains_Natoms.map(line => line._2);
		}

	var t0 = System.nanoTime()
	/** Parse PDB */	
	var PDBfile= new ParsePDB(args(0))
	var chain_coor = PDBfile.chain_coor.cache();
	var Chains_Natoms= PDBfile.Chains_Natoms;
	var Chains = PDBfile.Chains;
	var Natoms = PDBfile.Natoms;

	/** Get Relevant Indices For Final Matrix*/
	var Mat_indices:Array[Int]=Array();
	for(i <- 0 until (Natoms.collect().length +1))
   	{Mat_indices = Mat_indices:+ Natoms.collect().take(i).sum ; }
     
	var Total_atoms=0;
	var cutoff=args(1).toInt;
	var gamma=args(2);
	val mb = 1024*1024;
	
	/** Define Null Coordinate Matrix*/
	var coo_matrix_input:org.apache.spark.rdd.RDD[(Long, Long, Double)] = sc.parallelize(Array[(Long, Long, Double)]()).cache();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Build Graph laplacian
	/** Enumerate Over A Chain*/
	for ((chain,count1) <- Chains.collect().zipWithIndex){
		var nrow=Natoms.collect()(count1);
		Total_atoms += nrow;
		
		/** Get All Chain Combinations*/
		var combinations = Chains.cartesian(Chains).filter{ case (x,y) => x==chain & x<=y };
		
		for ((comb,count2) <- combinations.collect().zipWithIndex){
			println(comb);
			
			/** Build Diagonal Kirchhoff */
			if(comb._1==comb._2){
				var coord=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4));		
				var Coord_ind = coord.zipWithIndex
				var Combs=Coord_ind.cartesian(Coord_ind);
				
				/** Compute RMSD*/
				var rdd_cartesian=Combs.map{ case (((x1, y1,z1),id1), ((x2, y2,z2),id2)) => (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)))};
				var rdd2=rdd_cartesian.map{x => if (x._3 < cutoff) (x._1, x._2,-gamma.toDouble) else (x._1, x._2,0.toDouble)};
				var rdd3= rdd2.filter{x => (x._3!= 0)};

				//update row and column indices
				rdd3= rdd3.map{case(i,j,v)=> (i + Mat_indices(count1),j+Mat_indices(count2 + count1),v) };
				
				/** Get Coordinate Matrix Format*/
				coo_matrix_input = coo_matrix_input.union(rdd3);			
				}

			else {
				/** Build Non-Diagonal Kirchhoff */
				var coord1=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4)).zipWithIndex;
				var coord2=chain_coor.filter(chain_coor=> chain_coor._1==comb._2).map(x=> (x._2,x._3,x._4)).zipWithIndex;
				var Combs=coord1.cartesian(coord2);

				var rdd_cartesian=Combs.map{ case (((x1, y1,z1),id1), ((x2, y2,z2),id2)) => (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)))};
				var rdd2=rdd_cartesian.map{x => if (x._3 < cutoff) (x._1, x._2,-gamma.toDouble) else (x._1, x._2,0.toDouble)};
				var rdd3= rdd2.filter{x => (x._3!= 0)};

				rdd3= rdd3.map{case(i,j,v)=> (i + Mat_indices(count1),j+Mat_indices(count2 + count1),v) };
				coo_matrix_input= coo_matrix_input.union(rdd3);						
			}	
	}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/** Enumerate Over A Chain*/
	//Build Hessian
	/*for ((chain,count1) <- Chains.collect().zipWithIndex){
		var nrow=Natoms.collect()(count1);
		Total_atoms += nrow;
		
		/** Get All Chain Combinations*/
		var combinations = Chains.cartesian(Chains).filter{ case (x,y) => x==chain & x<=y };
		
		for ((comb,count2) <- combinations.collect().zipWithIndex){
			println(comb);
			
			/** Build Diagonal Kirchhoff */
			if(comb._1==comb._2){
				var coord=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4));		
				var Coord_ind = coord.zipWithIndex
				var Combs=Coord_ind.cartesian(Coord_ind);
				
				/** Compute RMSD*/
				var rdd_cartesianO=Combs.map{ case (((x1, y1,z1),id1), ((x2, y2,z2),id2)) => (id1, id2, Array((x2-x1)*(x2-x1),(x2-x1)*(y2-y1),(x2-x1)*(z2-z1), (y2-y1)*(x2-x1),(y2-y1)*(y2-y1),(y2-y1)*(z2-z1), (z2-z1)*(x2-x1),(z2-z1)*(y2-y1),(z2-z1)*(z2-z1) ))};				
				var rdd_cartesian=Combs.map{ case (((x1, y1,z1),id1), ((x2, y2,z2),id2)) => (Array(3*id1+0, 3*id2+0,(x2-x1)*(x2-x1)), Array(3*id1+0, 3*id2+1,(x2-x1)*(y2-y1)),Array(3*id1+0, 3*id2+2,(x2-x1)*(z2-z1)), Array(3*id1+1, 3*id2+0,(y2-y1)*(x2-x1)) , Array(3*id1+1, 3*id2+1,(y2-y1)*(y2-y1)), Array(3*id1+1, 3*id2+2,(y2-y1)*(z2-z1)) , Array(3*id1+2, 3*id2+0,(z2-z1)*(x2-x1)) , Array(3*id1+2, 3*id2+1,(z2-z1)*(y2-y1)) , Array(3*id1+2, 3*id2+2,(z2-z1)*(z2-z1)))};				
				
				


				rdd3 = rdd3.union(rdd3);

				var rdd2=rdd_cartesian.map{x => if (x._3 < cutoff) (x._1, x._2,-gamma.toDouble) else (x._1, x._2,0.toDouble)};
				var rdd3= rdd2.filter{x => (x._3!= 0)};

				//update row and column indices
				rdd3= rdd3.map{case(i,j,v)=> (i + Mat_indices(count1),j+Mat_indices(count2 + count1),v) };
				
				/** Get Coordinate Matrix Format*/
				coo_matrix_input = coo_matrix_input.union(rdd3);			
				}

			else {
				/** Build Non-Diagonal Kirchhoff */
				var coord1=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4)).zipWithIndex;
				var coord2=chain_coor.filter(chain_coor=> chain_coor._1==comb._2).map(x=> (x._2,x._3,x._4)).zipWithIndex;
				var Combs=coord1.cartesian(coord2);

				var rdd_cartesian=Combs.map{ case (((x1, y1,z1),id1), ((x2, y2,z2),id2)) => (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)))};
				var rdd2=rdd_cartesian.map{x => if (x._3 < cutoff) (x._1, x._2,-gamma.toDouble) else (x._1, x._2,0.toDouble)};
				var rdd3= rdd2.filter{x => (x._3!= 0)};

				rdd3= rdd3.map{case(i,j,v)=> (i + Mat_indices(count1),j+Mat_indices(count2 + count1),v) };
				coo_matrix_input= coo_matrix_input.union(rdd3);						
			}	
	}
	}
    */
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	var coo_matrix_entry = coo_matrix_input.map(e => MatrixEntry(e._1, e._2, e._3));
	var coo_matrix_UT = new CoordinateMatrix(coo_matrix_entry);

	//Transpose coo_matrix
	var coo_matrix_LT = coo_matrix_UT.transpose;
	var coo_matrix_entries = coo_matrix_UT.entries.union(coo_matrix_LT.entries).distinct();
	var coo_matrix = new CoordinateMatrix(coo_matrix_entries)
	
	// Diagonalize RDD
	var diag_vals = sc.parallelize(coo_matrix.toRowMatrix.computeColumnSummaryStatistics.numNonzeros.toArray.map(_.toLong)).cache();
	var diag_indices_vals = diag_vals.zipWithIndex.map{case(x,y)=> (y,y,(x.toDouble)-1)};
	var diag_entries = diag_indices_vals.map(e => MatrixEntry(e._1, e._2, e._3));
	
	//Compute Kirchhoff Matrix
	var nondiag_entries = coo_matrix.entries.filter{case MatrixEntry(row,column,value) => row!=column};
	coo_matrix_entries = diag_entries.union(nondiag_entries);
	var Kirchhoff_coo_matrix = new CoordinateMatrix(coo_matrix_entries);
	println(Kirchhoff_coo_matrix.toIndexedRowMatrix.numRows())
	//Kirchhoff_coo_matrix.toIndexedRowMatrix.rows.collect().foreach(line => println(line));

	
	//Singular value decomposition
	var k = args(3).toInt; //N_singvalues
	val mat: RowMatrix = Kirchhoff_coo_matrix.toRowMatrix
	//val mat: IndexedRowMatrix = Kirchhoff_coo_matrix.toIndexedRowMatrix
	val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)
	//val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = false)
	//val svd: SingularValueDecomposition[IndexedRowMatrix, Matrix] = mat.computeSVD(k, computeU = true)
	val U: RowMatrix = svd.U // The U factor is a RowMatrix.
	//val U: IndexedRowMatrix = svd.U
	val s: Vector = svd.s // The singular values are stored in a local dense vector.
	val V: Matrix = svd.V //The V factor is a local dense matrix.
	println("ahaha")
	
	//println(s);
	
	//Save to a file
	val s1=s.toArray;
	val s2= sc.parallelize(s1);
	s2.coalesce(1).saveAsTextFile("EigenValues_4x6h");
	val v1=V.toArray;
	val v2= sc.parallelize(v1);
	v2.coalesce(1).saveAsTextFile("EigenVectors_V_4x6h");	
	
	
	var runtime = Runtime.getRuntime;
	var t4 = System.nanoTime()
	println("Elapsed time: " + (t4 - t0)/1000000000.0 + "s")
	System.out.println("New session1,total memory = %s, used memory = %s, free memory = %s".format(runtime.totalMemory/mb, (runtime.totalMemory - runtime.freeMemory) / mb, runtime.freeMemory/mb));
	println("System size: " + Natoms.sum() + "atoms")

}
}