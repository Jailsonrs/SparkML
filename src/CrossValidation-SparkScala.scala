//ML libraries
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.classification.LogisticRegression
//Evaluation libraries
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
//Validation libraries
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row

/* TODO:
	    - remove undesired columns from the dataset
	    - parse the datatypes to desired ones in a clever way
	    - do n-fold crossvalidation 
	  	- Print confusion matrix and other binary classification metrics

	Jailson - 03 - 2022
*/

object LogisticRegressionApp extends App {

	val droplist: List[String] =  List(

	"ticker", ASSET_CVRG_RATIO, CAPITAL_EXPEND,                 
	CAP_EXPEND_RATIO, CASH_RATIO, CFO_TO_AVG_CURRENT_LIABILITIES, 
	CF_CASH_FROM_OPER, CF_FREE_CASH_FLOW, CUR_RATIO,                      
	EARN_FOR_COMMON, EBITDA, EBITDA_TO_INTEREST_EXPN,        
	EBITDA_TO_REVENUE, EBIT_TO_INT_EXP, ENTERPRISE_VALUE,               
	EV_TO_T12M_SALES, FCF_TO_TOTAL_DEBT, GROSS_MARGIN,                   
	GROSS_PROFIT, HISTORICAL_MARKET_CAP, IS_DIL_EPS_CONT_OPS,            
	NET_DEBT, NET_DEBT_TO_CASHFLOW, NET_DEBT_TO_EBITDA,             
	NET_INCOME_TO_COMMON_MARGIN, QUICK_RATIO, SALES_GROWTH,                   
	SALES_REV_TURN,	SHORT_AND_LONG_TERM_DEBT, TOTAL_DEBT_TO_EV,               
	TOT_DEBT_TO_COM_EQY, TOT_DEBT_TO_TOT_ASSET
	)  

	// Loads data.
	val path = "/test-files/data/amzn.csv"

	val df = spark.read.format("csv")
		.option("sep", ";")
		.option("inferSchema", "true")
		.option("header", "true")
		.load(path)

	val df3 = spark.read
		.option("delimiter", ";")
		.option("header", "true")
		.csv(path)

	val lr = new LogisticRegression().setMaxIter(10)

	// Setting up Grid for crossvalidation
	val grid_glm = newParamGridBuild()
		.addGrid(lr.regParamm, 
				 Array(0.1 0.01,0.001,0.0001,0.8,0.9,1,))
		.build()

	val cv = new CrossValidator()
		.setEstimator(lr)
		.setEvaluator(BinaryClassificationEvaluator)
		.setEstimatorParamMaps(paramGrid)
		.setNumFolds(6)
		.setFamily("binomial")

	println(s"Binomial coefficients: ${mlrModel.coefficientMatrix}")
	println(s"Binomial intercepts: ${mlrModel.interceptVector}")

	
}