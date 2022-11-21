#include "CudaLBFGS/lbfgs.h"
#include "CudaLBFGS/error_checking.h"
#include "CudaLBFGS/timer.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

//#include <device_functions.h>

using namespace std;

class cpu_sqtrack_chi2 : public cpu_cost_function
{
public:
	cpu_sqtrack_chi2(size_t n_points, float* h_drift, float* h_res, 
			float* h_p1x, float* h_p1y, float* h_p1z, 
			float* h_deltapx, float* h_deltapy, float* h_deltapz) 
	: cpu_cost_function(4),
	m_n_points (n_points),
	m_h_drift (h_drift),
	m_h_res (h_res),
	m_h_p1x (h_p1x),
	m_h_p1y (h_p1y),
	m_h_p1z (h_p1z),
	m_h_deltapx (h_deltapx),
	m_h_deltapy (h_deltapy),
	m_h_deltapz (h_deltapz){}

	void cpu_f_gradf(const floatdouble* h_x, floatdouble *h_chi2, floatdouble *h_grad)
	{
		for (size_t n = 0; n<m_numDimensions; n++)
		{
			h_grad[n] = 0.0;
		}
		*h_chi2 = 0.0;
		for(size_t i = 0; i<m_n_points; i++){
			const float Den = sqrtf( m_h_deltapy[i] * m_h_deltapy[i] +  m_h_deltapx[i] * m_h_deltapx[i] + 2.0f * m_h_deltapx[i] * m_h_deltapy[i] * h_x[2] * h_x[3] );
			const float dca = ( -1.0f * m_h_deltapy[i] * ( m_h_p1x[i] - h_x[0] ) + m_h_deltapx[i] * ( m_h_p1y[i] - h_x[1] ) - m_h_p1z[i] * ( h_x[2] * m_h_deltapy[i] - h_x[3] * m_h_deltapy[i] )  ) / Den;
			//const float value = ( h_drift[i] - dca ) / h_res[i];
			*h_chi2+= ( m_h_drift[i] - dca )*( m_h_drift[i] - dca ) / m_h_res[i] / m_h_res[i];
			
			h_grad[0]+= -2.0f * ( m_h_drift[i] - dca ) * m_h_deltapy[i] / Den / m_h_res[i] / m_h_res[i];
			h_grad[1]+= 2.0f * ( m_h_drift[i] - dca ) * m_h_deltapx[i] / Den / m_h_res[i] / m_h_res[i];
			h_grad[2]+= -2.0f * ( m_h_drift[i] - dca ) * ( m_h_deltapy[i] * m_h_p1z[i] / Den + dca * h_x[3] * m_h_deltapx[i] * m_h_deltapy[i] / Den / Den ) / m_h_res[i] / m_h_res[i];	
			h_grad[3]+= -2.0f * ( m_h_drift[i] - dca ) * ( -1.0f * m_h_deltapy[i] * m_h_p1z[i] / Den + dca * h_x[2] * m_h_deltapx[i] * m_h_deltapy[i] / Den / Den ) / m_h_res[i] / m_h_res[i];	
		}
	}

private:

	size_t m_n_points;
	float *m_h_drift;
	float *m_h_res;
	float *m_h_p1x;
	float *m_h_p1y;
	float *m_h_p1z;
	float *m_h_deltapx;
	float *m_h_deltapy;
	float *m_h_deltapz;
};

namespace gpu_sqtrack_chi2_d
{
//	__device__ size_t m_n_points;
//	__device__ float* m_drift;
//	__device__ float* m_res;
//	__device__ float* m_p1x;
//	__device__ float* m_p1y;
//	__device__ float* m_p1z;
//	__device__ float* m_deltapx;
//	__device__ float* m_deltapy;
//	__device__ float* m_deltapz;

	__global__ void kernel_combined_f_gradf(const size_t n_points, const float* d_drift, const float* d_res, 
		const float* d_p1x, const float* d_p1y, const float* d_p1z, 
		const float* d_deltapx, const float* d_deltapy, const float* d_deltapz, 
		const float* d_x, float& d_chi2, float* d_grad)
	{
		d_chi2 = 0.0f;
		d_grad[0] = 0.0f;
		d_grad[1] = 0.0f;
		d_grad[2] = 0.0f;
		d_grad[3] = 0.0f;
		//printf("%d %1.6f %1.6f %1.6f %1.6f \n", n_points, d_x[0], d_x[1], d_x[2], d_x[3]);
		for(size_t i = 0; i<6; i++){
			printf("%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f %1.6f %1.6f \n", d_drift[i], d_res[i], d_p1x[i], d_p1y[i], d_p1z[i], d_deltapx[i], d_deltapy[i], d_deltapz[i]);
			const float Den = sqrtf( d_deltapy[i] * d_deltapy[i] * (1.0f+d_x[2]*d_x[2]) + d_deltapx[i] * d_deltapx[i] * (1.0f+d_x[3]*d_x[3]) - 2.0f * d_deltapx[i] * d_deltapy[i] * d_x[2] * d_x[3] );
			const float dca = (-d_deltapy[i]*(d_p1x[i] - d_x[0]) + d_deltapx[i]*(d_p1y[i] - d_x[1]) + d_p1z[i]*(d_x[2]*d_deltapy[i] - d_x[3]*d_deltapx[i]) ) / Den;
			
			//const float value = ( d_drift[i] - dca ) / d_res[i];
			d_chi2+= ( d_drift[i] - dca )*( d_drift[i] - dca ) / d_res[i] / d_res[i];
			printf("den = %1.6f dca = %1.6f chi2+ = %1.6f\n", Den, dca, ( d_drift[i] - dca )*( d_drift[i] - dca ) / d_res[i] / d_res[i]);

			d_grad[0]+= +2.0f * ( d_drift[i] - dca ) * d_deltapy[i] / Den / d_res[i] / d_res[i];
			d_grad[1]+= -2.0f * ( d_drift[i] - dca ) * d_deltapx[i] / Den / d_res[i] / d_res[i];
			d_grad[2]+= -2.0f * ( d_drift[i] - dca ) * ( -1.0f * d_deltapy[i] * d_p1z[i] / Den + dca * d_x[3] * d_deltapx[i] * d_deltapy[i] / Den / Den ) / d_res[i] / d_res[i];	
			d_grad[3]+= -2.0f * ( d_drift[i] - dca ) * ( d_deltapx[i] * d_p1z[i] / Den + dca * d_x[2] * d_deltapx[i] * d_deltapy[i] / Den / Den ) / d_res[i] / d_res[i];	
		}
		
		printf("chi2 = %1.6f\n", d_chi2);
		d_chi2 = d_chi2/0.;
	}
}

class gpu_sqtrack_chi2 : public cost_function
{
public:
	gpu_sqtrack_chi2(const size_t n_points, const float* d_drift, const float* d_res, 
			const float* d_p1x, const float* d_p1y, const float* d_p1z, 
			const float* d_deltapx, const float* d_deltapy, const float* d_deltapz) 
	: cost_function(4) 
	{
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_n_points, gpu_sqtrack_chi2_d::m_n_points) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_drift, gpu_sqtrack_chi2_d::m_drift) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_res, gpu_sqtrack_chi2_d::m_res) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_p1x, gpu_sqtrack_chi2_d::m_p1x) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_p1y, gpu_sqtrack_chi2_d::m_p1y) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_p1z, gpu_sqtrack_chi2_d::m_p1z) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_deltapx, gpu_sqtrack_chi2_d::m_deltapx) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_deltapy, gpu_sqtrack_chi2_d::m_deltapy) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_deltapz, gpu_sqtrack_chi2_d::m_deltapz) );

		//CudaSafeCall( cudaMalloc((void**)&m_n_points, sizeof(size_t)) );
		CudaSafeCall( cudaMalloc(&m_d_drift, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_res, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_p1x, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_p1y, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_p1z, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_deltapx, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_deltapy, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_deltapz, n_points * sizeof(float)) );
		
		//printf("%lu %lu \n", n_points, m_n_points);
		CudaSafeCall( cudaMemcpy(&m_n_points, &n_points, sizeof(size_t), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_drift, d_drift, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_res, d_res, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_p1x, d_p1x, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_p1y, d_p1y, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_p1z, d_p1z, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_deltapx, d_deltapx, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_deltapy, d_deltapy, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_deltapz, d_deltapz, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		//printf("%lu %lu \n", n_points, m_n_points);
	}
	
	//h_x is what we want to minimize!
	void f_gradf(const float* d_x, float *d_chi2, float *d_gradf)
	{
		
		gpu_sqtrack_chi2_d::kernel_combined_f_gradf<<<1, 1>>>(m_n_points, m_d_drift, m_d_res, 
									m_d_p1x, m_d_p1y, m_d_p1z, 
									m_d_deltapx, m_d_deltapy, m_d_deltapz,
			 						d_x, *d_chi2, d_gradf);
		CudaCheckError();
	}
		
private:
	
	size_t m_n_points;
	float *m_d_drift;
	float *m_d_res;
	float *m_d_p1x;
	float *m_d_p1y;
	float *m_d_p1z;
	float *m_d_deltapx;
	float *m_d_deltapy;
	float *m_d_deltapz;
};

int main(int argc, char **argv)
{

	size_t maxIter = 500;
	float gradientEps = 1e-6f;

	int n_points = 6;
	
	//float drift[6] = {0.000000,0.000000,0.000000,0.000000,0.000000,0.000000};
	//float res[6] = {0.601311,0.601311,0.583413,0.583413,0.583413,0.583413};
	//float p1x[6] = {97.094971,96.025970,118.072472,117.044777,76.127060,75.085052};
	//float p1y[6] = {-131.875977,-131.860977,-132.290207, -132.307205,-132.994751,-132.992752};
	//float p1z[6] = {1346.886841,1339.906860,1372.907227,1365.917236,1321.749268,1314.779175};
	//float deltapx[6] = {-0.711784,-0.711784,-66.804100,-66.804100,66.165398,66.165398};
	//float deltapy[6] = {261.799988,261.799988,261.772003,261.772003,261.934998,261.934998};
	//float deltapz[6] = {0.673091,0.673091,-0.574838,-0.574838,-0.045143,-0.045143};
	float drift[6] = {0.000000,0.000000,0.000000,0.000000,0.000000,0.000000};
	float res[6] = {0.601311,0.601311,0.583413,0.583413,0.583413,0.583413};
	float p1x[6] = {-107.038040,-106.024055,-63.163315,-62.107838,-148.856369,-147.815201};
	float p1y[6] = {-132.431320,-132.410660,-133.177597,-133.184402,-132.442978,-132.446091};
	float p1z[6] = {1347.024414,1340.042969,1372.482910,1365.497803,1322.266968,1315.291992};
	//-0.711784,-0.711784,-66.804100,-66.804100,66.165398,66.165398};
	//float deltapy[6] = {261.799988,261.799988,261.772003,261.772003,261.934998,261.934998};
	//float deltapz[6] = {0.673091,0.673091,-0.574838,-0.574838,-0.045143,-0.045143};
	float deltapx[6] = {-0.7182,-0.7182,-67.33,-67.33,66.686,66.686};
	float deltapy[6] = {264.16, 264.16, 263.83, 263.83, 264.0, 264.0};
	float deltapz[6] = {0.67915, 0.67915, -0.57936, -0.57936, -0.045499,-0.045499};
	
	gpu_sqtrack_chi2 p1(n_points, drift, res, p1x, p1y, p1z, deltapx, deltapy, deltapz);

	lbfgs minimizer(p1);
	minimizer.setMaxIterations(maxIter);
	minimizer.setGradientEpsilon(gradientEps);
	
	//float x[4] = {0, 0.0, 0.0, 0.0};
	float x[4] = {94.919998, -50.00, -0.14992, 0.074924};

	float *d_x;
	CudaSafeCall( cudaMalloc(&d_x,   4 * sizeof(float)) );
	CudaSafeCall( cudaMemcpy(d_x, x, 4 * sizeof(float), cudaMemcpyHostToDevice) );
	
	lbfgs::status stat2 = minimizer.minimize(d_x);
	
	cout << lbfgs::statusToString(stat2).c_str() << endl;

	CudaSafeCall( cudaMemcpy(x, d_x, 4 * sizeof(float), cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaFree(d_x) );

	cout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << endl;

	
	return 0;
}



