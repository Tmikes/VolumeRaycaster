#version 330
// Input vertex data, different for all executions of this shader.
in vec2 vertexPosition_modelspace;
in vec2 vertexUV;

// Output data ; will be interpolated for each fragment.
out vec2 UV;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main() {

	// Output position of the vertex, in clip space : MVP * position
	gl_Position = MVP * vec4(vertexPosition_modelspace.x, vertexPosition_modelspace.y, 0, 1);

	// UV of the vertex. No special space for this one.
	UV = vertexUV;
}