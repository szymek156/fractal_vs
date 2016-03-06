#version 330 core

//
// Interpolated values from 
// the vertex shader.
//

in vec2 uv;

uniform sampler2D sampler;

//
// Ouput data.
//

out vec4 color;

void main()
{	 
  color = texture(sampler, uv); 
}