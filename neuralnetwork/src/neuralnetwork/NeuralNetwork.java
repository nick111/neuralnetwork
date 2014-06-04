package neuralnetwork;

import java.util.List;


public class NeuralNetwork {
	static String learnFileAddress = "file/learnIris.txt";
	static String testFileAddress = "file/testIris.txt";
	static String setosa ="setosa";
	static String versicolor = "versicolor";
	static String virginica = "virginica";
	

	public static void main(String[] args) {
		

		//  ������
		 Value value = new Value();
		 value.build();
		
		 
		//  �w�K
		//   �w�K�̏I�������Ƃ��Č덷�̂Q�悪�ǂ͈̔͂Ɏ��܂�ׂ������`����
		 double cc = 0.3;
		//   �덷�̒�`
		 double E;
		 // �w�K�t�@�C���̓ǂݍ���
		 List<String[]> learnData = FileUtil.getStringsByQuot(learnFileAddress);
		 // 1��̃��[�v��
		 int maxLoop = (int)2e+3;
		 
		//   �덷�����������Ɏ��܂�܂Ŋw�K���J��Ԃ�
		 do{
			 E = 0;
			 
			 // �y�f�o�b�O�p�z
			 int seikaiSuh = 0;
			 
			 for(int i = 0; i < maxLoop; i++){

				 // �����_���ɓ��̓f�[�^��I��
				 int x = (int) (Math.random() * learnData.size());
				 
				 // ���̓f�[�^����͑w�ɓ����
				 value.inputData(learnData.get(x));
				 
				 // forward����
				 value.forward();
				 
				// �y�w�K�f�[�^�z�o�͂̉񓚂𓾂�B
				 double[] ans = value.getAns();
				 String ansIris = getIrisName(ans);
				 if(ansIris.equals(learnData.get(x)[4])){
					 seikaiSuh++;
				 };
				 
				 // ���t�M����ǂݍ���
				 double[] teachSignal = getTeachSignal(learnData.get(x)[4]);
				 
				 
				 // backward����
				 value.backward(teachSignal);
				 
				 // �d�݂𒲐߂���i�w�K�j
				 value.learn();
				 
				 // �덷�̂Q��a�𑫂��i���[�v����o���Ƃ��Ƀ��[�v�̌����Ŋ���A�P���[�v���ς̌덷�̂Q��a���o���j
				 E += value.eval(teachSignal);

			 }
			 // �덷�̂Q��a�̕��ς��v�Z����
			 E /= maxLoop;
			 
			 // �y�w�K�f�[�^�p�z����
			 System.out.println("�w�K�f�[�^" + maxLoop + "���A" + seikaiSuh + "����");
			 
		 }while(E > cc);
		 
		 
		//  ����
		//  ���؂̐��������o��

		 // �e�X�g�t�@�C���̓ǂݍ���
		 List<String[]> testData = FileUtil.getStringsByQuot(testFileAddress);
		 
		 // ���𐔂̒�`
		 int numOfRightAns = 0;
		 
		 for(int i = 0; i < testData.size(); i++){
			 System.out.println(i + "�ڂ̃f�[�^");
			 for(int k = 0; k < testData.get(i).length -1; k++){
				 System.out.println(testData.get(i)[k]);
			 }
			 System.out.println();
			 
			 // ���̓f�[�^����͑w�ɓ����
			 value.inputData(testData.get(i));
			 
			 // forward����
			 value.forward();
			 
			 // �o�͂̉񓚂𓾂�B
			 double[] ans = value.getAns();
			 String ansIris = getIrisName(ans);
			 
			 // �������ʂ��o��
			 System.out.println("�\�z��" + ansIris);
			 System.out.println("���ʂ�" + testData.get(i)[testData.get(i).length - 1]);
			 
			 
			 if(ansIris.equals(testData.get(i)[testData.get(i).length - 1])){
				 System.out.println("�����I");
				 numOfRightAns++;
			 }else{
				 System.out.println("�s����...");
			 }
			 
		 }
		 System.out.println(testData.size() + "�̃f�[�^�̂����A" + numOfRightAns +"�̐����I");
		 
	}
	
	// ���̖��O���狳�t�M���̔z���Ԃ�
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
	
	// ���ʂ�����̖��O�𓾂�
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
