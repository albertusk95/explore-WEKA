package explore;

import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.converters.ArffLoader.ArffReader;

import weka.core.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;

import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;


public class TaskTucil02 {
	
	// objek dataset awal
	static Instances train_data;
	
	// objek dataset yang sudah difilter
	static Instances filtered_train_data;
	
	// menyimpan objek data baru (berupa dataset) yang akan diprediksi
    static Instances new_instances;
	
	// objek filter
	static Discretize filter;
	
	// objek classifier yang menggunakan filter
	static FilteredClassifier classifier;
	
	//static ArrayList<Double> text_input = new ArrayList<>();
	
	// menyimpan input dari user untuk setiap atribut
	static ArrayList<String> text_input = new ArrayList<>();

	
	/**
	* Solution for Task 1
	* Membaca dataset
	*/
	protected static void loadDataSet(String fileName) {
		
		try {
		
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				
				// membaca file dengan format ARFF
				ArffReader arff = new ArffReader(reader);
				
				// menyimpan isi file (bagian training data) dalam variabel trainData
				train_data = arff.getData();
				
				// bagian data yang akan diprediksi nilainya adalah atribut terakhir
				train_data.setClassIndex(train_data.numAttributes() - 1);
			
				// output pesan yang menandakan proses pembacaan file berhasil
				System.out.println("[OK] Akses dataset" + "\n");
				System.out.println("Lokasi dataset: " + fileName + "\n");
				System.out.println("---------------------------------------------------" + "\n");
				
				reader.close();
		
		} catch (IOException e) {
			
				// kasus jika pembacaan file gagal
				System.out.println("[FAIL] Gagal mengakses dataset: " + fileName + "\n");
				System.out.println("--------------------------------------------------------------------" + "\n");
		}
	}
	
	/**
	* Solution for Task 2
	* Mengaplikasikan filter (discretize for supervised learning)
	*/
	protected static void applyFilter() {
		
		try {
		
			// setup filter
			filter = new Discretize();
			filter.setInputFormat(train_data);

			// apply filter
			filtered_train_data = Filter.useFilter(train_data, filter);
		
		} catch (Exception e) {
			
			// kasus jika prosedur evaluasi gagal
			System.out.println("[FAIL] Error dalam mengimplementasikan filter" + "\n");
			System.out.println("--------------------------------------------------------------------" + "\n");
	
		}
		
	}
	
	/**
	* Solution for Task 3
	* Melakukan pembelajaran dataset dengan skema 10-fold cross validation
	*/
	protected static void learn_kCrossValidation() {
		
		try {
		
			// penentuan jenis classifier
			classifier = new FilteredClassifier();
			classifier.setFilter(filter);
			classifier.setClassifier(new NaiveBayes());
			
			Random rand = new Random(1);
			Instances randData = new Instances(filtered_train_data);
			randData.randomize(rand);
			
			int folds = 10;
			
			if (randData.classAttribute().isNominal()) {
				randData.stratify(folds);
			}

			Evaluation eval = new Evaluation(randData);
		  
			for (int n = 0; n < folds; n++) {
				
				// digunakan oleh StratifiedRemoveFolds filter
				Instances train = randData.trainCV(folds, n);
				Instances test = randData.testCV(folds, n);
				
				// digunakan oleh Explorer/Experimenter
				//Instances train = randData.trainCV(folds, n, rand);

				// membangun and mengevaluasi classifier
				classifier.buildClassifier(train);
				eval.evaluateModel(classifier, test);
			}

			// menampilkan hasil evaluasi model
			System.out.println("[OK] Evaluasi classifier" + "\n");
			System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
			System.out.println("=========================" + "\n");
			System.out.println(eval.toClassDetailsString() + "\n");
			System.out.println("--------------------------------------------------------------------" + "\n");
		
		} catch (Exception e) {
				
				// kasus jika prosedur evaluasi gagal
				System.out.println("[FAIL] Error dalam melakukan evaluasi k-fold cross validation" + "\n");
				System.out.println("--------------------------------------------------------------------" + "\n");
		
		}
		
	}
	
	
	/**
	* Solution for Task 4
	* Melakukan pembelajaran dataset dengan skema full-training
	*/
	protected static void learn_fullTraining() {
		
		try {
		
				// penentuan jenis classifier
				classifier = new FilteredClassifier();
				classifier.setFilter(filter);
				classifier.setClassifier(new NaiveBayes());

				// prosedur pembentukan model dari dataset yang dipilih
				classifier.buildClassifier(filtered_train_data);
					            
				// output pesan sukses pembentukan model
				System.out.println("[OK] Pelatihan & Pembentukan model oleh classifier full training" + "\n");
				System.out.println("--------------------------------------------------------------------" + "\n");
			
		} catch (Exception e) {
				
				// kasus jika proses pembentukan model (training) gagal
				System.out.println("[FAIL] Error dalam pembentukan model full training" + "\n");
				System.out.println("--------------------------------------------------------------------" + "\n");
				
		}
	
	}
	
	
	/**
	* Solution for Task 5
	* Menyimpan (save) model/hipotesis hasil pembelajaran ke sebuah file eksternal
	*/
	protected static void savemodel(String scheme, String fileName) {
		
		try {
		
			// memasukkan model yang terbentuk ke dalam file yang dipilih
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
			out.writeObject(classifier);
			out.close();

			// output pesan sukses penyimpanan model ke dalam file
			System.out.println("[OK] Menyimpan model " + scheme + "\n");
			System.out.println("Lokasi penyimpanan: " + fileName + "\n");
			System.out.println("------------------------------------------------------------------------" + "\n");
			
		} catch (IOException e) {
			
			// kasus jika prosedur penyimpanan model gagal
			System.out.println("[FAIL] Gagal menyimpan model " + scheme + " ke: " + fileName + "\n");
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		}
		
	}
	
	
	/**
	* Solution for Task 6
	* Membaca model dari file eksternal
	*/
	protected static void readModel(String fileName) {
		
		try {
		
			// validasi dan load model
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
			Object savedModel = in.readObject();
			classifier = (FilteredClassifier) savedModel;
			in.close();

			// output pesan sukses pembacaan file model
			System.out.println("[OK] Membaca file model" + "\n");
			System.out.println("Lokasi file model: " + fileName + "\n");
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		} catch (Exception e) {
			
			// kasus jika pembacaan file model gagal
			System.out.println("[FAIL] Gagal membaca file model dari: " + fileName + "\n");
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		}
		
	}
	
	
	/** 
	* Solution for Task 7
	* Membuat instance baru sesuai masukan dari pengguna untuk setiap nilai atribut
	* Masukan diasumsikan berasal dari file eksternal
	*/
	protected static void loadUserInput(String fileName) {
		
		try {
			
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			
			// mengambil isi file input
			String line;
			//text_input = "";
			
			while ((line = reader.readLine()) != null) {
				//text_input = text_input + " " + line;
				//text_input.add(Double.parseDouble(line));
				text_input.add(line);
			}
			
			reader.close();

			// output pesan sukses pembacaan file input
			System.out.println("[OK] Membaca file input" + "\n");
			System.out.println("Lokasi file input: " + fileName + "\n");
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		} catch (IOException e) {
			
			// kasus jika pembacaan file input gagal
			System.out.println("[FAIL] Gagal membaca file input dari: " + fileName + "\n");
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		}
		
	}
		
	/**
	* Membentuk instance baru sesuai masukan pengguna untuk 
	* setiap atribut
	*/	
	protected static void createUserInstance() {

		try {
			/** 
			* Membentuk nilai atribut yang akan diprediksi (class).
			* Nilai tersebut disimpan sementara dalam sebuah ArrayList.
			*/
			ArrayList<String> isi_attr_class = new ArrayList<>(3);
			
			//List isi_attr_class = new ArrayList(3);
			isi_attr_class.add("Iris-setosa");
			isi_attr_class.add("Iris-versicolor");
			isi_attr_class.add("Iris-virginica");

			/**
			* Membentuk objek atribut yang terdiri dari lima buah,
			* yaitu class (yang akan diprediksi) dan atribut lainnya (sepallength, sepalwidth, dll)
			*/
			/*
			Attribute attr1 = new Attribute("sepallength",(ArrayList<Double>) null);
			Attribute attr2 = new Attribute("sepalwidth",(ArrayList<Double>) null);
			Attribute attr3 = new Attribute("petallength",(ArrayList<Double>) null);
			Attribute attr4 = new Attribute("petalwidth",(ArrayList<Double>) null);
			Attribute attr5 = new Attribute("class", isi_attr_class);
			*/
			
			/*
			Attribute attr1 = new Attribute("sepallength");
			Attribute attr2 = new Attribute("sepalwidth");
			Attribute attr3 = new Attribute("petallength");
			Attribute attr4 = new Attribute("petalwidth");
			Attribute attr5 = new Attribute("class", isi_attr_class);
			*/
			
			//"\"(-inf-5.55]\""
			ArrayList<String> isi_attr_sepallength = new ArrayList<>();
			isi_attr_sepallength.add("'(-inf-5.55]'");
			isi_attr_sepallength.add("'(5.55-6.15]'");
			isi_attr_sepallength.add("'(6.15-inf)'");

			//"\"(-inf-2.95]\""
			ArrayList<String> isi_attr_sepalwidth = new ArrayList<>();
			isi_attr_sepalwidth.add("'(-inf-2.95]'");
			isi_attr_sepalwidth.add("'(2.95-3.35]'");
			isi_attr_sepalwidth.add("'(3.35-inf)'");

			//"\"(-inf-2.45]\""
			ArrayList<String> isi_attr_petallength = new ArrayList<>();
			isi_attr_petallength.add("'(-inf-2.45]'");
			isi_attr_petallength.add("'(2.45-4.75]'");
			isi_attr_petallength.add("'(4.75-inf)'");

			//"\"(-inf-0.8]\""
			ArrayList<String> isi_attr_petalwidth = new ArrayList<>();
			isi_attr_petalwidth.add("'(-inf-0.8]'");
			isi_attr_petalwidth.add("'(0.8-1.75]'");
			isi_attr_petalwidth.add("'(1.75-inf)'");

			Attribute attr1 = new Attribute("sepallength", isi_attr_sepallength);
			Attribute attr2 = new Attribute("sepalwidth", isi_attr_sepalwidth);
			Attribute attr3 = new Attribute("petallength", isi_attr_petallength);
			Attribute attr4 = new Attribute("petalwidth", isi_attr_petalwidth);
			Attribute attr5 = new Attribute("class", isi_attr_class);
			
			
			
			/*
			Attribute attr1 = new Attribute("sepallength", {"\"(-inf-5.55]\"","\"(5.55-6.15]\"","\"(6.15-inf)\""});
			Attribute attr2 = new Attribute("sepalwidth", {"\"(-inf-2.95]\"","\"(2.95-3.35]\"","\"(3.35-inf)\""});
			Attribute attr3 = new Attribute("petallength", {"\"(-inf-2.45]\"","\"(2.45-4.75]\"","\"(4.75-inf)\""});
			Attribute attr4 = new Attribute("petalwidth", {"\"(-inf-0.8]\"","\"(0.8-1.75]\"","\"(1.75-inf)\""});
			Attribute attr5 = new Attribute("class", isi_attr_class);
			*/
			
			/**
			* Menyimpan objek atribut yang sudah terbentuk ke dalam ArrayList
			* untuk sementara
			*/
			ArrayList<Attribute> attrObj = new ArrayList<>();
			attrObj.add(attr1);
			attrObj.add(attr2);
			attrObj.add(attr3);
			attrObj.add(attr4);
			attrObj.add(attr5);

			
			/**
			* Membentuk objek yang berupa dataset baru dengan nama relasi "DataTestIris",
			* menginisiasi atribut mana yang akan diprediksi (atribut terakhir), dan 
			* menambahkan input pengguna dari file input ke dalam atribut lainnya.
			*
			* NB:
			* Instances(java.lang.String name, FastVector attInfo, int capacity)
			* Creates an empty set of instances.
			*/
			new_instances = new Instances("DataTestIris", attrObj, 1);           
			new_instances.setClassIndex(4);
			
			DenseInstance instance = new DenseInstance(5);
			instance.setValue(attr1, text_input.get(0));
			instance.setValue(attr2, text_input.get(1));
			instance.setValue(attr3, text_input.get(2));
			instance.setValue(attr4, text_input.get(3));
			
			
			new_instances.add(instance);

			
			/*
			filter = new Discretize();
			filter.setInputFormat(filtered_train_data);

			// apply filter
			new_instances = Filter.useFilter(new_instances, filter);
			*/
			
			// output pesan sukses pembentukan objek atribut dan data
			System.out.println("[OK] Membentuk objek atribut dan data" + "\n \n");
			System.out.println(new_instances.toString());
			System.out.println("\n \n");
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		} catch (Exception e) {
			System.out.println("[FAIL] Gagal membentuk objek atribut dan data");
			System.out.println(e);
		}
	}

	/**
	* Solution for Task 8
	* Menentukan klasifikasi dari input user
	*/
	public static void inputClassification() {
		try {
		
			// mendapatkan nilai peluang masing-masing kelas
			double[] prediction_value = classifier.distributionForInstance(new_instances.instance(0));
		
			// klasifikasi teks input dengan atribut yang akan diprediksi adalah atribut pertama
			double txtClassified = classifier.classifyInstance(new_instances.instance(0));
			
			// output hasil klasifikasi teks input
			System.out.println("predicted-class: " + new_instances.classAttribute().value((int) txtClassified));
			System.out.println("prediction: " + prediction_value[(int) txtClassified]);
			
			// output pesan sukses prosedur klasifikasi teks input
			System.out.println("\n[OK] Klasifikasi input user" + "\n");
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		} catch (Exception e) {
	
			// kasus jika prosedur klasifikasi gagal
			System.out.println("[FAIL] Gagal klasifikasi input user" + "\n");
			System.out.println(e);
			System.out.println("------------------------------------------------------------------------" + "\n");
		
		}		
	}
	
	
	// For EXPERIMENT
	protected static void saveFilter(String fileName) {
		try {
			BufferedWriter  writer;

			writer = new BufferedWriter(new FileWriter(fileName));
			writer.write(filtered_train_data.toString());
			writer.newLine();
			writer.flush();
			writer.close();
		} catch (Exception e) {
			System.out.println(e);
		}
	}
	
	
	/**
	* MAIN function
	* @param args[0] - lokasi dataset
	* @param args[1] - lokasi saved model
	*/
	public static void main(String[] args) throws Exception {
		
		// Task 1 - Membaca dataset yang diberikan (iris.arff)
		loadDataSet(args[0]);
		
		// Task 2 - Mengaplikasikan filter yang mengubah tipe atribut, misalnya Discretize atau NumericToNominal
		applyFilter();
		
		
		//EXPERIMENT
		//saveFilter("../data/saved_filter.arff");
		//EXPERIMENT

		
		// Task 3 - Melakukan pembelajaran dataset dengan skema 10-fold cross validation
		learn_kCrossValidation();
		
		// Task 5 - Menyimpan (save) model/hipotesis hasil pembelajaran ke sebuah file eksternal
		savemodel("10-Fold-CV", args[1]);
		
		// Task 4 - Melakukan pembelajaran dataset dengan skema full-training
		learn_fullTraining();
		
		// Task 5 - Menyimpan (save) model/hipotesis hasil pembelajaran ke sebuah file eksternal
		savemodel("Full Training", args[2]);
		
		// Task 6 - Membaca (read) model/hipotesis dari file eksternal
		
		// k-fold CV
		readModel(args[1]);
		
		// Task 7 - Membuat instance baru sesuai masukan dari pengguna untuk setiap nilai atribut
		loadUserInput(args[3]);
		createUserInstance();
		
		// Task 8 - Melakukan klasifikasi dengan memanfaatkan model/hipotesis dan instance sesuai masukan pengguna pada no. 9
		inputClassification();
		
	}
}