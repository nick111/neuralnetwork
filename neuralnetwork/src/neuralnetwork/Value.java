package neuralnetwork;

public class Value {

	// 必要な変数
	//  レイヤーの層の数
	public int numOfLayer = 3;
	
	//  入力層の持つ素子の数（固定値の素子を含まない）
	public int numOfInDim = 4;
	
	//  中間層の持つ素子の数（固定値の素子を含まない）
	public int numOfCentDim = 4;
	
	//  出力層の持つ素子の数
	public int numOfOutDim = 3;
	
	//  素子の配列（２次元配列）
	public Neuron[][] neurons = new Neuron[numOfLayer][];
	
	//  ニューロン間の係数配列（３次元配列）
	public double[][][] weights = new double[numOfLayer][][];
	
	// 重み調節の係数(1/100)
	public double eps = 1e-2;
	
	public void build(){
		
		// 素子の配列に素子を入れる
		// まずは入力層（固定ニューロンも入れるのでプラス１する）
		neurons[0] = new Neuron[numOfInDim + 1];
		// シグモイドニューロンを入れる
		for(int i = 0; i < numOfInDim; i++){
			neurons[0][i] = new SigmoidNeuron();
		}
		// 固定値ニューロンを入れる
		neurons[0][numOfInDim] = new FixedNeuron();
		
		// 次に中間層の素子の配列に素子を入れる
		neurons[1] = new Neuron[numOfCentDim + 1];
		for(int i = 1; i < numOfLayer - 1; i++){
			for(int k = 0; k < numOfCentDim; k++){
				neurons[i][k] = new SigmoidNeuron();
			}
			neurons[i][numOfCentDim] = new FixedNeuron();
		}
		
		// 最後に出力層の素子の配列に素子を入れる（この層は固定値不要）
		neurons[numOfLayer - 1] = new Neuron[numOfOutDim];
		for(int i = 0; i < numOfOutDim; i++){
			neurons[numOfLayer - 1][i] = new SigmoidNeuron();
		}
		
		
		// 入力層の重みを乱数で初期化する(-1～1の間で重み付けする）
		weights[0] = new double[numOfInDim + 1][];
		for(int i = 0; i < numOfInDim + 1; i++){
			weights[0][i] = new double[numOfCentDim + 1];
			for(int k = 0; k < numOfCentDim + 1; k++){
				weights[0][i][k] = Math.random() * 2 + (-1);
			}
		}
		
		// 中間層が１層以上であれば、本当はここで中間層の最後の層以外の部分の重み付けをするが、
		// 今回は１層のみなので省略
		
		// 中間層の最後の層（出力層の１つ前の層）の重み付け
		weights[numOfLayer - 2] = new double[numOfCentDim + 1][];
		for(int i = 0; i < numOfCentDim + 1; i++){
			weights[numOfLayer - 2][i] = new double[numOfOutDim];
			for(int k = 0; k < numOfOutDim; k++){
				weights[numOfLayer - 2][i][k] = Math.random() * 2 + (-1);
			}
		}
		
	}
	
	// 入力ファイルのデータを入力層の素子にインプットする
	public void inputData(String[] data){
		for(int i = 0; i < numOfInDim; i++){
			neurons[0][i].in = Double.valueOf(data[i]);
		}
	}
	
	// forwardする
	public void forward(){
		// まず入力層の出力値を計算
		for(int i = 0; i < numOfInDim + 1; i++){
			neurons[0][i].calculate();
		}
		// 入力層から次の層への重み付き出力値を計算し、次の層の入力に加算する
		//  iが次の層、kが入力層をカウントする
		for(int i = 0; i < numOfCentDim; i++){
			double in = 0.0;
			for(int k = 0; k < numOfInDim + 1; k++){
				in += neurons[0][k].out * weights[0][k][i];
			}
			neurons[1][i].in = in;
		}
		
		// 【3層しかないので省略】中間層を次の層に持つ中間層の出力値を計算する
		// 【3層しかないので省略】中間層を次の層に持つ中間層の次の層への重み付き出力値を計算し、次の層の入力に加算する
		
		// 最後の中間層の出力値を計算
		for(int i = 0; i < numOfCentDim + 1; i++){
			neurons[numOfLayer - 2][i].calculate();
		}
		// 中間層から出力層への重み付き出力値を計算し、次の層の入力に加算する
		//  iが出力層、kがその前の層をカウントする
		for(int i = 0; i < numOfOutDim; i++){
			double in = 0.0;
			for(int k = 0; k < numOfCentDim + 1; k++){
				in += neurons[numOfLayer - 2][k].out * weights[numOfLayer - 2][k][i];
			}
			neurons[numOfLayer - 1][i].in = in;
		}
		
		// 出力層のoutを計算する
		for(int i = 0; i < numOfOutDim; i++){
			neurons[numOfLayer - 1][i].calculate();
		}		
		
	}
	
	public void backward(double[] teachSignal){
		// 教師信号から出力層のdE/dsの値をセットする
		for(int i = 0; i < numOfOutDim; i++){
			neurons[numOfLayer - 1][i].dEds = 2 * (neurons[numOfLayer - 1][i].out - teachSignal[i]) * neurons[numOfLayer - 1][i].df();
		}
		// 出力層の一つ前の層のdE/dsの値をセットする
		for(int i = 0; i < numOfCentDim + 1; i++){
			double value = 0.0;
			for(int k = 0; k < numOfOutDim; k++){
				value += neurons[numOfLayer - 1][k].dEds * weights[numOfLayer - 2][i][k];
			}
			neurons[numOfLayer - 2][i].dEds = value * neurons[numOfLayer - 2][i].df();
		}
		
		// 【例のごとく省略】中間層同士の連携は３層だからなし
		
		// 中間層から入力層へ伝播
		for(int i = 0; i < numOfInDim + 1; i++){
			double value = 0.0;
			for(int k = 0; k < numOfCentDim + 1; k++){
				value += neurons[1][k].dEds * weights[0][i][k];
			}
			neurons[0][i].dEds = value * neurons[0][i].df();
		}
		
		
	}
	
	public void learn(){
		// 入力層から中間層への重み調整
		for(int i = 0; i < numOfInDim + 1; i++){
			for(int k = 0; k < numOfCentDim + 1; k++){
				weights[0][i][k] -= eps * neurons[1][k].dEds * neurons[0][i].out;
			}
		}
		// 中間層から出力層への重み調整
		for(int i = 0; i < numOfCentDim + 1; i++){
			for(int k = 0; k < numOfOutDim; k++){
				weights[numOfLayer - 2][i][k] -= eps * neurons[numOfLayer - 1][k].dEds * neurons[numOfLayer - 2][i].out;
			}
		}
	}
	
	// 出力層の誤差の２乗和を出力する
	public double eval(double[] teachSignal){
		double e = 0.0;
		for(int i = 0; i < numOfOutDim; i++){
			e += Math.pow(teachSignal[i] - neurons[numOfLayer - 1][i].out,2);
		}
		return e;
	}
	
	// 答えを返す
	public double[] getAns(){
		int k = 0;
		for(int i = 1; i < numOfOutDim; i++){
			if(neurons[numOfLayer - 1][i].out > neurons[numOfLayer - 1][k].out){
				k = i;
			}
		}
		double[] returnVal = new double[numOfLayer];
		for(int i = 0; i < numOfOutDim; i++){
			if(i == k){
				returnVal[i] = 1.0;
			}else{
				returnVal[i] = 0.0;
			}
		}
		return returnVal;
	}
	
}
