package neuralnetwork;

public class Value {

	// �K�v�ȕϐ�
	//  ���C���[�̑w�̐�
	public int numOfLayer = 3;
	
	//  ���͑w�̎��f�q�̐��i�Œ�l�̑f�q���܂܂Ȃ��j
	public int numOfInDim = 4;
	
	//  ���ԑw�̎��f�q�̐��i�Œ�l�̑f�q���܂܂Ȃ��j
	public int numOfCentDim = 4;
	
	//  �o�͑w�̎��f�q�̐�
	public int numOfOutDim = 3;
	
	//  �f�q�̔z��i�Q�����z��j
	public Neuron[][] neurons = new Neuron[numOfLayer][];
	
	//  �j���[�����Ԃ̌W���z��i�R�����z��j
	public double[][][] weights = new double[numOfLayer][][];
	
	// �d�ݒ��߂̌W��(1/100)
	public double eps = 1e-2;
	
	public void build(){
		
		// �f�q�̔z��ɑf�q������
		// �܂��͓��͑w�i�Œ�j���[�����������̂Ńv���X�P����j
		neurons[0] = new Neuron[numOfInDim + 1];
		// �V�O���C�h�j���[����������
		for(int i = 0; i < numOfInDim; i++){
			neurons[0][i] = new SigmoidNeuron();
		}
		// �Œ�l�j���[����������
		neurons[0][numOfInDim] = new FixedNeuron();
		
		// ���ɒ��ԑw�̑f�q�̔z��ɑf�q������
		neurons[1] = new Neuron[numOfCentDim + 1];
		for(int i = 1; i < numOfLayer - 1; i++){
			for(int k = 0; k < numOfCentDim; k++){
				neurons[i][k] = new SigmoidNeuron();
			}
			neurons[i][numOfCentDim] = new FixedNeuron();
		}
		
		// �Ō�ɏo�͑w�̑f�q�̔z��ɑf�q������i���̑w�͌Œ�l�s�v�j
		neurons[numOfLayer - 1] = new Neuron[numOfOutDim];
		for(int i = 0; i < numOfOutDim; i++){
			neurons[numOfLayer - 1][i] = new SigmoidNeuron();
		}
		
		
		// ���͑w�̏d�݂𗐐��ŏ���������(-1�`1�̊Ԃŏd�ݕt������j
		weights[0] = new double[numOfInDim + 1][];
		for(int i = 0; i < numOfInDim + 1; i++){
			weights[0][i] = new double[numOfCentDim + 1];
			for(int k = 0; k < numOfCentDim + 1; k++){
				weights[0][i][k] = Math.random() * 2 + (-1);
			}
		}
		
		// ���ԑw���P�w�ȏ�ł���΁A�{���͂����Œ��ԑw�̍Ō�̑w�ȊO�̕����̏d�ݕt�������邪�A
		// ����͂P�w�݂̂Ȃ̂ŏȗ�
		
		// ���ԑw�̍Ō�̑w�i�o�͑w�̂P�O�̑w�j�̏d�ݕt��
		weights[numOfLayer - 2] = new double[numOfCentDim + 1][];
		for(int i = 0; i < numOfCentDim + 1; i++){
			weights[numOfLayer - 2][i] = new double[numOfOutDim];
			for(int k = 0; k < numOfOutDim; k++){
				weights[numOfLayer - 2][i][k] = Math.random() * 2 + (-1);
			}
		}
		
	}
	
	// ���̓t�@�C���̃f�[�^����͑w�̑f�q�ɃC���v�b�g����
	public void inputData(String[] data){
		for(int i = 0; i < numOfInDim; i++){
			neurons[0][i].in = Double.valueOf(data[i]);
		}
	}
	
	// forward����
	public void forward(){
		// �܂����͑w�̏o�͒l���v�Z
		for(int i = 0; i < numOfInDim + 1; i++){
			neurons[0][i].calculate();
		}
		// ���͑w���玟�̑w�ւ̏d�ݕt���o�͒l���v�Z���A���̑w�̓��͂ɉ��Z����
		//  i�����̑w�Ak�����͑w���J�E���g����
		for(int i = 0; i < numOfCentDim; i++){
			double in = 0.0;
			for(int k = 0; k < numOfInDim + 1; k++){
				in += neurons[0][k].out * weights[0][k][i];
			}
			neurons[1][i].in = in;
		}
		
		// �y3�w�����Ȃ��̂ŏȗ��z���ԑw�����̑w�Ɏ����ԑw�̏o�͒l���v�Z����
		// �y3�w�����Ȃ��̂ŏȗ��z���ԑw�����̑w�Ɏ����ԑw�̎��̑w�ւ̏d�ݕt���o�͒l���v�Z���A���̑w�̓��͂ɉ��Z����
		
		// �Ō�̒��ԑw�̏o�͒l���v�Z
		for(int i = 0; i < numOfCentDim + 1; i++){
			neurons[numOfLayer - 2][i].calculate();
		}
		// ���ԑw����o�͑w�ւ̏d�ݕt���o�͒l���v�Z���A���̑w�̓��͂ɉ��Z����
		//  i���o�͑w�Ak�����̑O�̑w���J�E���g����
		for(int i = 0; i < numOfOutDim; i++){
			double in = 0.0;
			for(int k = 0; k < numOfCentDim + 1; k++){
				in += neurons[numOfLayer - 2][k].out * weights[numOfLayer - 2][k][i];
			}
			neurons[numOfLayer - 1][i].in = in;
		}
		
		// �o�͑w��out���v�Z����
		for(int i = 0; i < numOfOutDim; i++){
			neurons[numOfLayer - 1][i].calculate();
		}		
		
	}
	
	public void backward(double[] teachSignal){
		// ���t�M������o�͑w��dE/ds�̒l���Z�b�g����
		for(int i = 0; i < numOfOutDim; i++){
			neurons[numOfLayer - 1][i].dEds = 2 * (neurons[numOfLayer - 1][i].out - teachSignal[i]) * neurons[numOfLayer - 1][i].df();
		}
		// �o�͑w�̈�O�̑w��dE/ds�̒l���Z�b�g����
		for(int i = 0; i < numOfCentDim + 1; i++){
			double value = 0.0;
			for(int k = 0; k < numOfOutDim; k++){
				value += neurons[numOfLayer - 1][k].dEds * weights[numOfLayer - 2][i][k];
			}
			neurons[numOfLayer - 2][i].dEds = value * neurons[numOfLayer - 2][i].df();
		}
		
		// �y��̂��Ƃ��ȗ��z���ԑw���m�̘A�g�͂R�w������Ȃ�
		
		// ���ԑw������͑w�֓`�d
		for(int i = 0; i < numOfInDim + 1; i++){
			double value = 0.0;
			for(int k = 0; k < numOfCentDim + 1; k++){
				value += neurons[1][k].dEds * weights[0][i][k];
			}
			neurons[0][i].dEds = value * neurons[0][i].df();
		}
		
		
	}
	
	public void learn(){
		// ���͑w���璆�ԑw�ւ̏d�ݒ���
		for(int i = 0; i < numOfInDim + 1; i++){
			for(int k = 0; k < numOfCentDim + 1; k++){
				weights[0][i][k] -= eps * neurons[1][k].dEds * neurons[0][i].out;
			}
		}
		// ���ԑw����o�͑w�ւ̏d�ݒ���
		for(int i = 0; i < numOfCentDim + 1; i++){
			for(int k = 0; k < numOfOutDim; k++){
				weights[numOfLayer - 2][i][k] -= eps * neurons[numOfLayer - 1][k].dEds * neurons[numOfLayer - 2][i].out;
			}
		}
	}
	
	// �o�͑w�̌덷�̂Q��a���o�͂���
	public double eval(double[] teachSignal){
		double e = 0.0;
		for(int i = 0; i < numOfOutDim; i++){
			e += Math.pow(teachSignal[i] - neurons[numOfLayer - 1][i].out,2);
		}
		return e;
	}
	
	// ������Ԃ�
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
