#version 330 core

//
// Non-transformed vertex
// coordinates. This var-
// iable has by default
// layout(location = 0).
//

in vec3 vertexPosition;

//
// This variable has by default
// layout(location = 1). Because is declared
// as second one.
//

in vec2 vertexUV;

//
// This will be input of fragment 
// shader.
//

out vec2 uv;

uniform mat4 MVP;

void main()
{
  vec4 v = vec4(vertexPosition, 1);
 
  gl_Position = MVP * v;
  
  uv = vertexUV; 
}

