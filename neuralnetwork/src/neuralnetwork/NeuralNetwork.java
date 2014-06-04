package neuralnetwork;

import java.util.List;


public class NeuralNetwork {
	static String learnFileAddress = "file/learnIris.txt";
	static String testFileAddress = "file/testIris.txt";
	static String setosa ="setosa";
	static String versicolor = "versicolor";
	static String virginica = "virginica";
	

	public static void main(String[] args) {
		

		//  初期化
		 Value value = new Value();
		 value.build();
		
		 
		//  学習
		//   学習の終了条件として誤差の２乗がどの範囲に収まるべきかを定義する
		 double cc = 0.3;
		//   誤差の定義
		 double E;
		 // 学習ファイルの読み込み
		 List<String[]> learnData = FileUtil.getStringsByQuot(learnFileAddress);
		 // 1回のループ回数
		 int maxLoop = (int)2e+3;
		 
		//   誤差が収束条件に収まるまで学習を繰り返す
		 do{
			 E = 0;
			 
			 // 【デバッグ用】
			 int seikaiSuh = 0;
			 
			 for(int i = 0; i < maxLoop; i++){

				 // ランダムに入力データを選ぶ
				 int x = (int) (Math.random() * learnData.size());
				 
				 // 入力データを入力層に入れる
				 value.inputData(learnData.get(x));
				 
				 // forwardする
				 value.forward();
				 
				// 【学習データ】出力の回答を得る。
				 double[] ans = value.getAns();
				 String ansIris = getIrisName(ans);
				 if(ansIris.equals(learnData.get(x)[4])){
					 seikaiSuh++;
				 };
				 
				 // 教師信号を読み込む
				 double[] teachSignal = getTeachSignal(learnData.get(x)[4]);
				 
				 
				 // backwardする
				 value.backward(teachSignal);
				 
				 // 重みを調節する（学習）
				 value.learn();
				 
				 // 誤差の２乗和を足す（ループから出たときにループの件数で割り、１ループ平均の誤差の２乗和を出す）
				 E += value.eval(teachSignal);

			 }
			 // 誤差の２乗和の平均を計算する
			 E /= maxLoop;
			 
			 // 【学習データ用】正解数
			 System.out.println("学習データ" + maxLoop + "中、" + seikaiSuh + "個正解");
			 
		 }while(E > cc);
		 
		 
		//  検証
		//  検証の成功率を出す

		 // テストファイルの読み込み
		 List<String[]> testData = FileUtil.getStringsByQuot(testFileAddress);
		 
		 // 正解数の定義
		 int numOfRightAns = 0;
		 
		 for(int i = 0; i < testData.size(); i++){
			 System.out.println(i + "個目のデータ");
			 for(int k = 0; k < testData.get(i).length -1; k++){
				 System.out.println(testData.get(i)[k]);
			 }
			 System.out.println();
			 
			 // 入力データを入力層に入れる
			 value.inputData(testData.get(i));
			 
			 // forwardする
			 value.forward();
			 
			 // 出力の回答を得る。
			 double[] ans = value.getAns();
			 String ansIris = getIrisName(ans);
			 
			 // 推測結果を出力
			 System.out.println("予想は" + ansIris);
			 System.out.println("結果は" + testData.get(i)[testData.get(i).length - 1]);
			 
			 
			 if(ansIris.equals(testData.get(i)[testData.get(i).length - 1])){
				 System.out.println("正解！");
				 numOfRightAns++;
			 }else{
				 System.out.println("不正解...");
			 }
			 
		 }
		 System.out.println(testData.size() + "個のデータのうち、" + numOfRightAns +"個の正解！");
		 
	}
	
	// ゆりの名前から教師信号の配列を返す
	public static double[] getTeachSignal(String irisName){
		double[] teachSignal = new double[3];
		if(irisName.equals(setosa)){
			teachSignal[0] = 1.0;
			teachSignal[1] = 0.0;
			teachSignal[2] = 0.0;
		}else if(irisName.equals(versicolor)){
			teachSignal[0] = 0.0;
			teachSignal[1] = 1.0;
			teachSignal[2] = 0.0;
		}else if(irisName.equals(virginica)){
			teachSignal[0] = 0.0;
			teachSignal[1] = 0.0;
			teachSignal[2] = 1.0;
		}
		return teachSignal;
		
	}
	
	// 結果からゆりの名前を得る
	public static String getIrisName(double[] ans){
		String returnVal ="";
		if(ans[0] == 1.0){
			returnVal = setosa;
		}else if(ans[1] == 1.0){
			returnVal = versicolor;
		}else if(ans[2] == 1.0){
			returnVal = virginica;
		}
		return returnVal;
	}
	

}
