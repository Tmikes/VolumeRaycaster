
// Input vertex data, different for all executions of this shader.
attribute vec2 vertexPosition_modelspace;
attribute vec2 vertexUV;
attribute vec2 vertexCenter;

// Output data ; will be interpolated for each fragment
varying vec2 UV;
flat varying vec2 center;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main() {

	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(vertexPosition_modelspace.x, vertexPosition_modelspace.y, 0, 1);

	// UV of the vertex. No special space for this one.
	UV = vertexUV;
	center = vertexCenter;
}