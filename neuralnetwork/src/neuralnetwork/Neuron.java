package neuralnetwork;

public abstract class Neuron {

	// 前のレイヤーから入力された値
	public double in;
	
	// 次のレイヤーへ出力する値
	public double out;
	
	// dE/dsの値（次のレイヤーからもらってくる）
	public double dEds;
	
	// 入力の総和を与えて出力値を判定するもの
	public abstract double f(double x);
	
	// fの微分。backwardのときに使用する
	public abstract double df();
	
	// 出力値を計算し格納する
	public void calculate() {
		out = f(in);
	}
	
}

// 固定値を出力するニューロン
class FixedNeuron extends Neuron {
	double fixedOut = 1.0;
	
	public double f(double x) {	
		return fixedOut;
	}
	
	public double df() {
		return 0;
	}
	
	
}


// シグモイド関数の値を出力するニューロン
class SigmoidNeuron extends Neuron {
	
	// シグモイド関数のxの係数
	double gain = 5.0;
	
	public double f(double x) {
		return 1/(1+Math.exp(-gain*x));
	}
	
	public double df() {
		return gain*out*(1-out);
	}
	
	
	
}