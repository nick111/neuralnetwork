package neuralnetwork;

public abstract class Neuron {

	// �O�̃��C���[������͂��ꂽ�l
	public double in;
	
	// ���̃��C���[�֏o�͂���l
	public double out;
	
	// dE/ds�̒l�i���̃��C���[���������Ă���j
	public double dEds;
	
	// ���͂̑��a��^���ďo�͒l�𔻒肷�����
	public abstract double f(double x);
	
	// f�̔����Bbackward�̂Ƃ��Ɏg�p����
	public abstract double df();
	
	// �o�͒l���v�Z���i�[����
	public void calculate() {
		out = f(in);
	}
	
}

// �Œ�l���o�͂���j���[����
class FixedNeuron extends Neuron {
	double fixedOut = 1.0;
	
	public double f(double x) {	
		return fixedOut;
	}
	
	public double df() {
		return 0;
	}
	
	
}


// �V�O���C�h�֐��̒l���o�͂���j���[����
class SigmoidNeuron extends Neuron {
	
	// �V�O���C�h�֐���x�̌W��
	double gain = 5.0;
	
	public double f(double x) {
		return 1/(1+Math.exp(-gain*x));
	}
	
	public double df() {
		return gain*out*(1-out);
	}
	
	
	
}