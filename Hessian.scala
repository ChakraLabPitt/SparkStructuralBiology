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
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Build Hessian
	
	for ((chain,count1) <- Chains.collect().zipWithIndex) {
		var nrow=Natoms.collect()(count1);
		Total_atoms += nrow;

		/** Get All Chain Combinations*/
		var combinations = Chains.cartesian(Chains).filter{ case (x,y) => x==chain & x<=y };

		for ((comb,count2) <- combinations.collect().zipWithIndex) {


			println(comb);
			//Build Diagonal Hessian
			if (comb._1==comb._2) {
				var coord=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4));
				var Coord_ind = coord.zipWithIndex.map {case (values, key) => (key, values)};
				var Combs=Coord_ind.cartesian(Coord_ind);

				/** Compute RMSD*/							
				var rdd_cartesian_O=Combs.map{ case ((id1,(x1, y1,z1)), (id2,(x2, y2,z2))) => (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)))};
				var rdd2=rdd_cartesian_O.filter{x => x._3 < cutoff};
				var rdd3= rdd2.filter{x => (x._3!= 0)};
				var rdd3_ind = rdd3.map{x => (x._1, x._2)};


				var joinReady_Combs = Combs.map{case (x,y) => ((x._1, y._1), (x._2,y._2))};
				var joinReady_rdd3_ind = rdd3_ind.map{ key => (key, ())};
				var joinReady_rmsd = rdd3.map{ key => ((key._1,key._2), (key._3*key._3))};

				var Combs_rel = joinReady_Combs.join(joinReady_rdd3_ind).mapValues(_._1);


				var rdd_cartesian = Combs_rel.map{ case (x,y) => (3*x._1+0, 3*x._2+0,(y._2._1-y._1._1)*(y._2._1-y._1._1)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}.union(Combs_rel.map{ case (x,y) => (3*x._1+0, 3*x._2+1,(y._2._1-y._1._1)*(y._2._2-y._1._2)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+0, 3*x._2+2,(y._2._1-y._1._1)*(y._2._3-y._1._3)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+1, 3*x._2+0,(y._2._2-y._1._2)*(y._2._1-y._1._1)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+1, 3*x._2+1,(y._2._2-y._1._2)*(y._2._2-y._1._2)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+1, 3*x._2+2,(y._2._2-y._1._2)*(y._2._3-y._1._3)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+2, 3*x._2+0,(y._2._3-y._1._3)*(y._2._1-y._1._1)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+2, 3*x._2+1,(y._2._3-y._1._3)*(y._2._2-y._1._2)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+2, 3*x._2+2,(y._2._3-y._1._3)*(y._2._3-y._1._3)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))});

				rdd3= rdd_cartesian.map{case(i,j,v)=> (i + 3*Mat_indices(count1),j+3*Mat_indices(count2 + count1),v)};
				
				/** Get Coordinate Matrix Format*/
				coo_matrix_input = coo_matrix_input.union(rdd3);
				}

			else {
				/** Build Non-Diagonal Hessian */
				var coord1=chain_coor.filter(chain_coor=> chain_coor._1==comb._1).map(x=> (x._2,x._3,x._4)).zipWithIndex.map {case (values, key) => (key, values)};
				var coord2=chain_coor.filter(chain_coor=> chain_coor._1==comb._2).map(x=> (x._2,x._3,x._4)).zipWithIndex.map {case (values, key) => (key, values)};
				var Combs=coord1.cartesian(coord2);

				var rdd_cartesian_O=Combs.map{ case ((id1,(x1, y1,z1)), (id2, (x2, y2,z2))) => (id1, id2, math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)))};
				
				var rdd2=rdd_cartesian_O.filter{x => x._3 < cutoff};
				var rdd3= rdd2.filter{x => (x._3!= 0)};

				var rdd3_ind = rdd3.map{x => (x._1, x._2)};

				//var joinReadyCoord_ind = Coord_ind.map {case (values, key) => (key, values)};
				var joinReady_Combs = Combs.map{case (x,y) => ((x._1, y._1), (x._2,y._2))};
				var joinReady_rdd3_ind = rdd3_ind.map{ key => (key, ())};

				var Combs_rel = joinReady_Combs.join(joinReady_rdd3_ind).mapValues(_._1);

				var rdd_cartesian = Combs_rel.map{ case (x,y) => (3*x._1+0, 3*x._2+0,(y._2._1-y._1._1)*(y._2._1-y._1._1)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}.union(Combs_rel.map{ case (x,y) => (3*x._1+0, 3*x._2+1,(y._2._1-y._1._1)*(y._2._2-y._1._2)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+0, 3*x._2+2,(y._2._1-y._1._1)*(y._2._3-y._1._3)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+1, 3*x._2+0,(y._2._2-y._1._2)*(y._2._1-y._1._1)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+1, 3*x._2+1,(y._2._2-y._1._2)*(y._2._2-y._1._2)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+1, 3*x._2+2,(y._2._2-y._1._2)*(y._2._3-y._1._3)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+2, 3*x._2+0,(y._2._3-y._1._3)*(y._2._1-y._1._1)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+2, 3*x._2+1,(y._2._3-y._1._3)*(y._2._2-y._1._2)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))}).union(Combs_rel.map{ case (x,y) => (3*x._1+2, 3*x._2+2,(y._2._3-y._1._3)*(y._2._3-y._1._3)/((y._2._1-y._1._1)*(y._2._1-y._1._1)+ (y._2._2-y._1._2)*(y._2._2-y._1._2)+(y._2._3-y._1._3)*(y._2._3-y._1._3)))});

				rdd3= rdd_cartesian.map{case(i,j,v)=> (i + 3*Mat_indices(count1),j+3*Mat_indices(count2 + count1),v)};
				
				/** Get Coordinate Matrix Format*/
				coo_matrix_input = coo_matrix_input.union(rdd3);

				} 
			}
		}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Transpose the matrix
	var coo_matrix_input_LT = coo_matrix_input.map{ case (i,j,k) => (j,i,k)};

	var coo_matrix_input_all = coo_matrix_input_LT.union(coo_matrix_input).distinct();
	var coo_matrix_entries = coo_matrix_input_all.map(e => MatrixEntry(e._1, e._2, e._3));

	
	
	// Diagonalize RDD

	var diag_entries_1 = coo_matrix_entries.filter{case MatrixEntry(row, col, value) => col%3 ==0}.map{case MatrixEntry(row, _, value) => (row, value)}.reduceByKey(_ + _).map{case (row,value) => MatrixEntry(row, 3*(row/3),-value )};
	var diag_entries_2 = coo_matrix_entries.filter{case MatrixEntry(row, col, value) => col%3 ==1}.map{case MatrixEntry(row, _, value) => (row, value)}.reduceByKey(_ + _).map{case (row,value) => MatrixEntry(row, 3*(row/3)+1,-value )};
	var diag_entries_3 = coo_matrix_entries.filter{case MatrixEntry(row, col, value) => col%3 ==2}.map{case MatrixEntry(row, _, value) => (row, value)}.reduceByKey(_ + _).map{case (row,value) => MatrixEntry(row, 3*(row/3)+2,-value )};

	var diag_entries = diag_entries_1.union(diag_entries_2).union(diag_entries_3);
	coo_matrix_entries  = coo_matrix_entries.union(diag_entries);
	var coo_matrix = new CoordinateMatrix(coo_matrix_entries);

	
	// //Singular value decomposition
	// var k = args(3).toInt; //N_singvalues
	// val mat: RowMatrix = coo_matrix.toRowMatrix
	// val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)
	// val U: RowMatrix = svd.U // The U factor is a RowMatrix.
	// val s: Vector = svd.s // The singular values are stored in a local dense vector.
	// val V: Matrix = svd.V //The V factor is a local dense matrix.
	

	// //Save to a file
	// val s1=s.toArray;
	// val s2= sc.parallelize(s1);
	// s2.coalesce(1).saveAsTextFile("EigenValues_4x6h");
	// val v1=V.toArray;
	// val v2= sc.parallelize(v1);
	// v2.coalesce(1).saveAsTextFile("EigenVectors_V_4x6h");	
	
	
	var runtime = Runtime.getRuntime;
	var t4 = System.nanoTime()
	println("Elapsed time: " + (t4 - t0)/1000000000.0 + "s")
	System.out.println("New session1,total memory = %s, used memory = %s, free memory = %s".format(runtime.totalMemory/mb, (runtime.totalMemory - runtime.freeMemory) / mb, runtime.freeMemory/mb));
	println("System size: " + Natoms.sum() + "atoms")

}
}