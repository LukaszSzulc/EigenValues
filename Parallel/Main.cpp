#include <iostream>
#include <omp.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <math.h>
using namespace std;
const int numberOfIterations = 800;

vector<vector<double > > CreateMatrix(const int n)
{
	vector<vector<double > > result(n) ;
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = vector<double>(n);
		for (int j = 0; j < result[i].size(); j++)
		{
			if (i == j)
			{
				result[i][j] = 1.0;
			}
			else
			{
				result[i][j] = 0.1;
			}
		}
	}
	return result;
}

void PrintMatrix(const vector<vector<double > > &matrix)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[i].size(); j++) 
		{
			cout << matrix[i][j] << " ";
		}

		cout << endl;
	}
}


vector<double> CreateVector(int n)
{
	vector<double> result(n);
	for (int i = 0; i < n; i++)
	{
		result[i] = i + 1;
	}

	return result;
}

void PrintVector(const vector<double> &vector)
{
	for (int i = 0; i < vector.size(); i++)
	{
		cout << vector.at(i) << " ";
	}

	cout << endl;
}

vector<double>& NormalizeVector(vector<double> &vector)
{
	double sum = 0.0;
	for (int i = 0; i < vector.size(); i++)
	{
		sum += vector.at(i) * vector.at(i);
	}

	sum = sqrt(sum);

	for (int i = 0; i < vector.size(); i++)
	{
		vector[i] /= sum;
	}

	return vector;
}


vector<double> MultiplyMatrixByVector(const vector<vector<double > > &matrix, const vector<double> &v)
{
	vector<double> result(v.size());
	for (int i = 0; i < matrix.size(); i++)
	{
		double sum = 0.0;
		for (int j = 0; j < matrix[i].size(); j++)
		{
			sum += matrix[i][j] * v[j];
		}

		result[i] = sum;
	}

	return result;
}

vector<double> MultiplyMatrixByVectorParallel(const vector<double> &v)
{

	vector<double> result(v.size());
	double sum = 0.0;
	int size = v.size();
#pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < size; i++) 
		{
			sum = 0.0;
			for (int j = 0; j < size; j++)
			{
				if (i == j)
				{
					sum += v[j];
				}
				else
				{
					sum += 0.1 * v[j];
				}
			}

			result[i] = sum;
		}
	return result;
}

vector<double> MultiplyMatrixByVector(const vector<double> &v)
{

	vector<double> result(v.size());
	
	for (int i = 0; i < v.size(); i++)
	{
		double sum = 0.0;
		for (int j = 0; j < v.size(); j++)
		{
			if (i == j)
			{
				sum += v[j];
			}
			else
			{
				sum += 0.1 * v[j];
			}
		}

		result[i] = sum;
	}

	return result;
}

double Dot(const vector<double> &v1, const vector<double> &v2)
{
	double sum = 0.0;
	for (int i = 0; i < v1.size(); i++)
	{
		sum += v1.at(i) * v2.at(i);
	}

	return sum;
}

double VectorNorm(const vector<double> vector)
{
	double sum = 0.0f;
	for (int i = 0; i < vector.size(); i++)
	{
		sum += vector.at(i);
	}

	return sum;
}

double FindEigSingleThread(int matrixSize)
{
	double eps = 1e-10;
	vector<double> vec = CreateVector(matrixSize);
	vector<double> b = NormalizeVector(vec);
	vector<double> a;
	double vectorNorm = 0.0;
	for (int i = 0; i <= numberOfIterations; i++)
	{
		double tmpNorm = 0.0;
		if (i % 2 == 0)
		{
			a = MultiplyMatrixByVector(b);
			NormalizeVector(a);
			tmpNorm = VectorNorm(a);
		}
		else
		{
			b = MultiplyMatrixByVector(a);
			NormalizeVector(b);
			tmpNorm = VectorNorm(b);
		}
	}

	vector<double> result;
	double eigenValue = 0.0;

	result = MultiplyMatrixByVector(a);
	eigenValue = Dot(result, a);
	return eigenValue;
}



double FindEigMultithread(int matrixSize)
{
	vector<double> vec = CreateVector(matrixSize);
	vector<double> b = NormalizeVector(vec);
	vector<double> a;
	{

			for (int i = 0; i <= numberOfIterations; i++)
			{
				if (i % 2 == 0)
				{
					a = MultiplyMatrixByVectorParallel(b);
					NormalizeVector(a);
				}
				else
				{
					b = MultiplyMatrixByVectorParallel(a);
					NormalizeVector(b);
				}
			}	
	}

	vector<double> result;
	double eigenValue = 0.0;

	result = MultiplyMatrixByVector(a);
	eigenValue = Dot(result, a);
	return eigenValue;
}

int main()
{	
	ofstream myfile;
	myfile.open("parallel.txt");
	for (int i = 1000; i <= 10000; i += 1000)
	{
		for (int j = 2; j <= 4; j++)
		{
			double start = omp_get_wtime();
			double eigen = FindEigMultithread(i);
			cout << i << "\t" << j << "\t" << eigen << "\t" << omp_get_wtime() - start << endl;
			myfile << i << "\t" << j << "\t" << eigen << "\t"<< omp_get_wtime() - start << endl;
		}
	}
	myfile.close();

	myfile.open("single.txt");

	for (int i = 1000; i <= 10000; i += 1000)
	{
		double start = omp_get_wtime();
		double eigen = FindEigSingleThread(i);
		cout << i << "\t" << 1 << "\t" << eigen << "\t" << omp_get_wtime() - start << endl;
		myfile << i << "\t" << 1 << "\t" << eigen << "\t" << omp_get_wtime() - start << endl;
	}

	myfile.close();
	system("pause");
	return EXIT_SUCCESS;
}