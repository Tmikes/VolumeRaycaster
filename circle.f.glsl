
// Interpolated values from the vertex shaders
varying vec2 UV;
flat varying vec2 center;
// Ouput data

// Values that stay constant for the whole mesh.
uniform sampler1D myTextureSampler;

void main() {

	// Output color = color of the texture at the specified UV
	//color = texture(myTextureSampler, UV).rgb;
	gl_FragColor = texture(myTextureSampler, center);
}