
typedef struct
{
	float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
	float3 o;   // origin
	float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each aposWorld.xs
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__device__ uint rgbaFloatToInt(float4 pRgba)
{
	pRgba.x = __saturatef(pRgba.x);   // clamp to [0.0, 1.0]
	pRgba.y = __saturatef(pRgba.y);
	pRgba.z = __saturatef(pRgba.z);
	pRgba.w = __saturatef(pRgba.w);
	return (uint(pRgba.w * 255) << 24) | (uint(pRgba.z * 255) << 16) | (uint(pRgba.y * 255) << 8) | uint(pRgba.x * 255);
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH, float density, float transferOffset, float3 dim, float3 ratio)
{
	const int maxSteps = 2000;
	const float tstep = 0.003f;
	const float opacityThreshold = 0.95f;
	const float3 boxMin = { -1,-1,-1 };// -ratio;
	const float3 boxMax = { 1,1,1 };//ratio;

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;



	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit) {
		d_output[y*imageW + x] = rgbaFloatToInt(make_float4(0, 0, 0, 0));
		return;
	}
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

										// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float3 light = make_float3(1, 1, 1);
	float3 lightdir = mul(c_invViewMatrix, normalize(make_float3(1, 1, 0)));

	float attenuation = 1.0f;
	float shininess = 10.0f;
	float3 specularlight;
	float diffuseCoeff = 0.5f;
	float specularCoeff = 0.3f;
	float ambiantCoeff = 0.2f;
	float invdot, specPow = 10;

	for (int i = 0; i<maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float3 posWorld = (0.5f*pos + 0.5f) / ratio;
		float sample = tex3D(volumeTex, posWorld.x, posWorld.y, posWorld.z);
		//sample *= 64.0f;    // scale for 10-bit data

		// lookup in transfer function texture
		float4 col = tex3D(transferTex, sample, transferOffset, 0);
		col.w *= density;

		float gradx = (tex3D(volumeTex, posWorld.x + 0.5f / dim.x, posWorld.y, posWorld.z) - tex3D(volumeTex, posWorld.x - 0.5f / dim.x, posWorld.y, posWorld.z)) / 2;
		float gradz = (tex3D(volumeTex, posWorld.x, posWorld.y, posWorld.z + 0.5f / dim.z) - tex3D(volumeTex, posWorld.x, posWorld.y, posWorld.z - 0.5f / dim.z)) / 2;
		float grady = (tex3D(volumeTex, posWorld.x, posWorld.y + 0.5f / dim.y, posWorld.z) - tex3D(volumeTex, posWorld.x, posWorld.y - 0.5f / dim.y, posWorld.z)) / 2;

		float4 n = make_float4(normalize(make_float3(gradx, grady, gradz)), 1.0f);
		//n = normalize(n);
		//n = mul4(c_invViewMatrix, n);
		float dotprod = lightdir.x * n.x + lightdir.y * n.y + lightdir.z * n.z; // lambertian

		dotprod = max(0.0f, dotprod);
		dotprod = min(1.0f, dotprod);

		float3 ambianlight = make_float3(col.x, col.y, col.z);
		float3 diffuselight = dotprod * make_float3(light.x *col.x, light.y *col.y, light.z*col.z);

		//float3 
		if (dotprod < 0)
		{
			specularlight = make_float3(0, 0, 0);
		}
		else {
			invdot = dot(reflect(-lightdir, make_float3(n.x, n.y, n.z)), eyeRay.d);
			//specularlight = attenuation* pow(max(0.0f, invdot), shininess)*make_float3(1, 1, 1);
		}

		diffuselight *= diffuseCoeff;
		specularlight = specularCoeff * pow(max(0.0f, invdot), specPow)*make_float3(1, 1, 1);


		//diffuselight *= col.w;
		ambianlight *= col.w; //* occlusion;
							  //specularlight *= col.w;
		float3 color = ambianlight + diffuselight + specularlight;
		color.x = fminf(color.x, 1);
		color.y = fminf(color.y, 1);
		color.z = fminf(color.z, 1);




		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x = color.x * col.w;
		col.y = color.y * col.w;
		col.z = color.z * col.w;
		// "over" operator for front-to-back blending
		sum = sum + col * (1.0f - sum.w);

		// eposWorld.xt early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}
	// write output color
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}
