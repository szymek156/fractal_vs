
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <Windows.h>
#include <chrono>
#include <iomanip>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "CommonUtil.h"
#include "CudaUtil.h"
#include "GLUtil.h"
#include "Fractal.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"

using namespace std;

#define DEBUG

static const double BOUNDUARY = 4.0;
static const int NON_CLIENT_AREA = 64;

int fpsCnt_ = 0;

GLuint vertexArrayId_;

GLuint vertexBuffer_;

GLuint colorBuffer_;

GLuint textureId_;

GLuint uvbuffer_;

unsigned char *texData = NULL;

GLuint programId_;

static int texWidth_;

static int texHeight_;

static double bounduary_ = 4.0;

static float zoomSpeed_ = 0.0f;

static int enableMouse_ = 0;

static float mouseSensitivity_ = 1;

static int computeFractal_ = 2;

static double mouseX_ = 0;
static double mouseY_ = 0;

static bool quad_ = 0;

static double windowWidth_  = 800;

static double windowHeight_ = 600 + NON_CLIENT_AREA;
//
// An array of 3 vectors which
// represents 3 vertices.
//

static const GLfloat square[] = {
   -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   -1.0f,  1.0f, 0.0f,

   -1.0f,  1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f, 1.0f, 0.0f
};

static const GLfloat uvCoords[] = {
   0.0f, 0.0f,
   1.0f, 0.0f,
   0.0f,  1.0f,

   0.0f,  1.0f,
   1.0f, 0.0f,
   1.0f, 1.0f
};

#define VERTEX_ATTRIB 0

#define COLOR_ATTRIB 1

#define TEX_ATTRIB 1


int GLUtil::getTexWidth()
{
  return texWidth_;
}

int GLUtil::getTexHeight()
{
  return texHeight_;
}

double GLUtil::getZoom()
{
  return BOUNDUARY / bounduary_;
}

double GLUtil::getBounduary()
{
	return bounduary_;
}

double GLUtil::getMouseX()
{
  return mouseX_;
}

double GLUtil::getMouseY()
{
  return -mouseY_;
}

bool GLUtil::getQuad()
{
	return quad_;
}

void GLUtil::loadShaders(const char *vertexFilepath, const char *fragmentFilepath)
{
  //
  // Create the shaders.
  //

  GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);

  GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

  //
  // Read the shaders code code from
  // the file.
  //

  string vertexShaderCode;

  ifstream vertexShaderStream(vertexFilepath, ios::in);

  if (vertexShaderStream.is_open())
  {
    string line = "";
    while (std::getline(vertexShaderStream, line))
    {
      vertexShaderCode += "\n" + line;
    }

    vertexShaderStream.close();
  }

  string fragmentShaderCode;

  ifstream fragmentShaderStream(fragmentFilepath, ios::in);

  if (fragmentShaderStream.is_open())
  {
    std::string Line = "";

    while (getline(fragmentShaderStream, Line))
    {
      fragmentShaderCode += "\n" + Line;
    }

    fragmentShaderStream.close();
  }

  GLint result = GL_FALSE;

  int infoLogLength;

  //
  // Compile.
  //

  #ifdef DEBUG
  cout << "loadShaders: Compiling shader: " << vertexFilepath << endl;
  cout << "loadShaders: Compiling shader: " << fragmentFilepath << endl;
  #endif

  char const *vertexSourcePointer = vertexShaderCode.c_str();

  glShaderSource(vertexShaderID, 1, &vertexSourcePointer, NULL);

  glCompileShader(vertexShaderID);

  //
  // Check Vertex Shader.
  //

  glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &result);

  glGetShaderiv(vertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);

  vector<char> vertexShaderErrorMessage(infoLogLength);

  glGetShaderInfoLog(vertexShaderID, infoLogLength, NULL, &vertexShaderErrorMessage[0]);

  #ifdef DEBUG
  cout << "loadShaders: Vertex shader result: "
       << &vertexShaderErrorMessage[0] << endl;
  #endif

  char const * fragmentSourcePointer = fragmentShaderCode.c_str();

  glShaderSource(fragmentShaderID, 1, &fragmentSourcePointer, NULL);

  glCompileShader(fragmentShaderID);

  //
  // Check Fragment Shader.
  //

  glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &result);

  glGetShaderiv(fragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);

  vector<char> fragmentShaderErrorMessage(infoLogLength);

  glGetShaderInfoLog(fragmentShaderID, infoLogLength, NULL,
      &fragmentShaderErrorMessage[0]);

  #ifdef DEBUG
  cout << "loadShaders: Vertex shader result: "
       << &vertexShaderErrorMessage[0] << endl; 
  #endif

  //
  // Create program.
  //

  #ifdef DEBUG
  cout << "loadShaders: Creating program." << endl;
  #endif

  GLuint programID = glCreateProgram();

  glAttachShader(programID, vertexShaderID);

  glAttachShader(programID, fragmentShaderID);

  glLinkProgram(programID);

  //
  // Check the program.
  //

  glGetProgramiv(programID, GL_LINK_STATUS, &result);

  glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);

  vector<char> programErrorMessage(std::max(infoLogLength, int(1)));

  glGetProgramInfoLog(programID, infoLogLength, NULL, &programErrorMessage[0]);

  #ifdef DEBUG
  cout << "loadShaders: Program creation result: "
       << &programErrorMessage[0] << endl;
  #endif

  glDeleteShader(vertexShaderID);

  glDeleteShader(fragmentShaderID);

  programId_ = programID;
}

void GLUtil::computeFPS()
{
  fpsCnt_++;

  stopWatch();

  if (getTimeDiffMs() > 1000)
  {
    char fps[256];
    sprintf(fps, "%d fps, iterations %d, zoom: %f",
                fpsCnt_, Fractal::getIterations(), getZoom());
    glutSetWindowTitle(fps);

    fpsCnt_ = 0;

    startWatch();
  }
}

void GLUtil::displayHandler()
{
  zoom();

  if (computeFractal_ > 0)
  {
	  std::chrono::time_point<std::chrono::system_clock>  start = std::chrono::system_clock::now();

	 // Fractal::mandelbrotGPUNative(1);

    Fractal::mandelbrotCPUNative(1);


	//Fractal::mandelbrotCPUNativeParallel(1);

    //Fractal::mandelbrotCPUQuad(1);

	std::chrono::time_point<std::chrono::system_clock> stop = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = (stop - start);

	cout << "Generated frame in " << std::fixed << std::setprecision(3) << elapsed_seconds.count() << "s\n";
    computeFractal_--;
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(programId_);

  // 1rst attribute buffer : vertices
  glEnableVertexAttribArray(VERTEX_ATTRIB);

  glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer_);

  //
  // This call tells the pipeline
  // how to interpret the data
  // inside the buffer.
  //

  glVertexAttribPointer(
     VERTEX_ATTRIB,      // must match the layout in the shader
     3,                  // size of one element (x, y, z)
     GL_FLOAT,           // type
     GL_FALSE,           // normalized?
     0,                  // offset to get another element
     (void*)0            // array buffer offset
  );

  glDrawArrays(GL_TRIANGLES, 0, 6 * 2 * 3);

  glDisableVertexAttribArray(VERTEX_ATTRIB);

//------------------------- texture --------------------------------//

  glEnableVertexAttribArray(TEX_ATTRIB);

  glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_);

  glVertexAttribPointer(
     TEX_ATTRIB,         // must match the layout in the shader
     2,                  // size of one element, (u, v)
     GL_FLOAT,           // type
     GL_FALSE,           // normalized?
     0,                  // stride
     (void*)0            // array buffer offset
  );

  glUseProgram(0);

  glutSwapBuffers();

  computeFPS();

  glutReportErrors();

}

void GLUtil::keyboardSpecialHandler(int key, int x, int y)
{
  #ifdef DEBUG
  cout << "keyboardSpecialHandler: key: '" << key
       <<  "' x: " << x << " y: " << y << endl;
  #endif

  double offset = bounduary_ / 4.0;

  switch (key)
  {
    case GLUT_KEY_LEFT:
    {
		mouseX_ -= offset;
      break;
    }
    case GLUT_KEY_UP:
    {
		mouseY_ -= offset;
      break;
    }
    case GLUT_KEY_RIGHT:
    {
		mouseX_ += offset;
      break;
    }
    case GLUT_KEY_DOWN:
    {
		mouseY_ += offset;
      break;
    }
  }

  computeFractal_ = 1;

}

void GLUtil::keyboardHandler(unsigned char key, int x, int y)
{
  #ifdef DEBUG
  cout << "keyboardHandler: key: '" << key
       <<  "' x: " << x << " y: " << y << endl;
  #endif

    switch (key)
    {
      case 27:
      {
        exit(EXIT_SUCCESS);
        break;
      }

	  case 'q':
	  {
		  quad_ = !quad_;
		  computeFractal_ = 1;
		  break;
	  }
      case ']':
      {
        zoomSpeed_ += 0.1;
        break;
      }
      case '[':
      {
        zoomSpeed_ -= 0.1;
        break;
      }
	  case ';':
      {
        Fractal::setIterations(Fractal::getIterations() - 500);

        computeFractal_ = 1;

        break;
      }
      case '\'':
      {
        Fractal::setIterations(Fractal::getIterations() + 500);

        computeFractal_ = 1;

        break;
      }
      case ' ':
      {
        enableMouse_ = !enableMouse_;

        break;
      }
      case 'z':
      {
        mouseSensitivity_ --;

        if (mouseSensitivity_ == 0)
        {
          mouseSensitivity_ = 1;
        }

        break;
      }
      case 'x':
      {
        mouseSensitivity_ ++;

        if (mouseSensitivity_ == 0)
        {
          mouseSensitivity_ = 1;
        }

        break;
      }
      case '0':
      {
        resetPosition();

        break;
      }
      case '1':
      {
        resetPosition();

		mouseX_ = -1.2583384664947936;

		mouseY_ = -0.032317669198187016;

        Fractal::setIterations(500);

        break;
      }
      case '2':
      {
        resetPosition();

		mouseX_ = -1.2487780999747029;

		mouseY_ = 0.071802096973029209;

        Fractal::setIterations(500);

        break;
      }
      case '3':
      {
        resetPosition();

		mouseX_ = -1.2583385189936513;

		mouseY_ = -0.032317635405726151;

        Fractal::setIterations(1500);

        break;
      }
      case '4':
      {
        resetPosition();

		mouseX_ = -1.2583384664947908;

		mouseY_ = -0.032317669198180785;

        Fractal::setIterations(500);

        break;
      }
      case '5':
      {
//        Real number:
//
//        -1.768,573,656,315,270,993,281,
//        742,915,329,544,712,934,120,053,405,549,882,
//        337,511,135,282,776,553,364,635,382,011,977,
//        933,536,332,198,647,808,795,874,576,643,230,
//        034,448,609,820,608,458,844,529,169,083,285,
//        379,260,833,581,131,961,323,480,667,495,949,
//        838,043,253,626,912,240,448,884,745,364,662,
//        832,495,906,454,3
//
//        imaginary number:
//
//        -0.000,964,296,851,358,280,000,176,242,720,
//        373,819,448,274,776,122,656,563,565,285,783,
//        153,307,047,554,366,655,893,028,615,382,795,
//        071,670,082,888,793,257,893,297,692,452,344,
//        749,770,824,889,473,425,648,018,389,868,316,
//        458,205,554,184,217,181,589,930,525,084,269,
//        263,834,905,711,879,329,676,832,512,425,574,
//        656,3

        resetPosition();

		//-1.47805321161304, -0.00295648593239646 fajne swiderki 
		//-1.4780532121658325, -0.0029564858414232731

		mouseX_ = -1.4780998580724920;

		mouseY_ = -0.0029962325962097328;

        Fractal::setIterations(500);

        break;
      }
      case '6':
      {
        resetPosition();

        mouseX_ = 0.3994999999000;

        mouseY_ = -0.195303;

        Fractal::setIterations(800);

        break;
      }
      case '7':
      {
        resetPosition();

        mouseX_ = -1.768611136076306;

        mouseY_ = -0.001266863985331;

        Fractal::setIterations(1300);

        break;
      }
      case '8':
      {
        resetPosition();

		mouseX_ = -1.7686112281079116;

		mouseY_ = -0.0012668963162883458;

        Fractal::setIterations(200);

        break;
      }
      default:
      {
        break;
      }
    }

    glutPostRedisplay();
}

void GLUtil::mouseHandler(int button, int state, int x, int y)
{
 // #ifdef DEBUG
  cout << "mouseHandler: button: " << button << " state: "
       << state << " x: " << x << " y: " << y << endl;
//  #endif

  switch (button)
  {
    case GLUT_LEFT_BUTTON:
    {
      if (state == GLUT_DOWN)
      {
		centerAt(x, y);
      }

      break;
    }
  }

  glutPostRedisplay();
}

void GLUtil::centerAt(int windowX, int windowY)
{
	//
	// Translate pointer coordinates to center
	// of the window.
	//

	double mouseX = (windowX - ((windowWidth_ / 2.0)));
	double mouseY = (windowY - ((windowHeight_ / 2.0)));

	double texWinWidthRatio = (texWidth_ / (double)windowWidth_);
	double texWinHeightRatio = (texHeight_/ (double)windowHeight_);

	int texX = mouseX * texWinWidthRatio;
	int texY = mouseY * texWinHeightRatio;

	double xRatio = bounduary_ / texWidth_;
	double yRatio = bounduary_ / texHeight_;

	mouseX_ += (texX * xRatio);
	mouseY_ += (texY * yRatio);

	cout << "windowX " << windowX << " windowY " << windowY << "\n";
	cout << "bounduary " << bounduary_ << " xRatio " << xRatio << " yRatio " << yRatio << " zoom x" << getZoom() << "\n";


	cout << "Centering at point (" << mouseX_ << ", " << mouseY_ << ")\n";

	computeFractal_ = 1;
}


/*
void GLUtil::centerAt(int x, int y)
{
	double ratioX = windowWidth_ / bounduary_;

	double ratioY = windowHeight_ / bounduary_;

	//
	// Translate pointer coordinates to center
	// of the window.
	//

	double mouseX = (x - ((windowWidth_ / 2.0)));
	double mouseY = (y - ((windowHeight_ / 2.0)));

	//
	// Scale coordinates to current region
	// boundaries for zoom 1 it is [-2, 2].
	//

	mouseX /= ratioX;
	mouseY /= ratioY;

	//
	// Translate according to last pivot.
	//

	mouseX_ += mouseX;
	mouseY_ += mouseY;
	
	cout << "in x " << x << " in y " << y << "\n";
	cout << "bounduary " << bounduary_ << " ratioX " << ratioX << " ratioY " << ratioY << " zoom x" << getZoom() << "\n";
	

	cout << "Centering at point (" << mouseX_ << ", " << mouseY_ << "\n";

	computeFractal_ = 1;
}
*/
/*
void GLUtil::passiveMouseHandler(int x, int y)
{
	static int old_x = 0;
	static int old_y = 0;

	if (x == old_x && y == old_y)
	{
		return;
	}

	old_x = x;
	old_y = y;

	cout << "in_x " << x << " in_y " << y << "\n";

	computeFractal_ = 1;
}
*/

void GLUtil::passiveMouseHandler(int x, int y)
{
}


void GLUtil::reshapeHandler(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void GLUtil::idleHandler()
{
  glutPostRedisplay();
}

int GLUtil::initWindow(int argc, char *argv[])
{
  glutInit(&argc, argv);

  //
  // Enable double buffering.
  //

  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(windowWidth_, windowHeight_);
  glutInitWindowPosition(0, 0);
  glutCreateWindow("Testo");

  //
  // Register handlers.
  //

  glutDisplayFunc(displayHandler);
  glutKeyboardFunc(keyboardHandler);
  glutSpecialFunc(keyboardSpecialHandler);
  glutMouseFunc(mouseHandler);
  glutPassiveMotionFunc(passiveMouseHandler);
  glutReshapeFunc(reshapeHandler);
  glutIdleFunc(idleHandler);

  glClearColor(0.3f, 0.3f, 1.0f, 1.0f);

  glewInit();

  if (!glewIsSupported( "GL_VERSION_2_0 " ))
  {
    cout << "initGL: ERROR! Required OpenGL extensions missing." << endl;

    return -1;
  }

  return 1;
}

void GLUtil::initObjects()
{
  //
  // Create objects.
  //

  glGenVertexArrays(1, &vertexArrayId_);

  glBindVertexArray(vertexArrayId_);

  glGenBuffers(1, &vertexBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(square), square, GL_STATIC_DRAW);

  //
  // Create texture.
  //

  glGenTextures(1, &textureId_);
  glBindTexture(GL_TEXTURE_2D, textureId_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texWidth_, texHeight_, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, NULL);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glUseProgram(programId_);

  GLuint samplerId  = glGetUniformLocation(programId_, "sampler");

  // Bind our texture in Texture Unit 0
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureId_);
  // Set our "myTextureSampler" sampler to user Texture Unit 0
  glUniform1i(samplerId, 0);

  glGenBuffers(1, &uvbuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uvCoords), uvCoords, GL_STATIC_DRAW);

  glutReportErrors();
}

//
// Set camera position etc.
//

void GLUtil::setScene()
{
  //
  // Projection matrix.
  //

  glm::mat4 projection = glm::perspective(
                             45.0f,       // fov
                             (float)(windowWidth_ / (float) (windowHeight_)), // aspect ratio
                             0.1f,        // near clip
                             100.0f       // far clip
                             );

  //
  // Camera matrix.
  //

  glm::mat4 view = glm::lookAt(
                       glm::vec3(0,0,2.0f), // position
                       glm::vec3(0,0,0),    // where to look
                       glm::vec3(0,1,0)     // up
                       );

  //
  // Model matrix.
  // Identity, don't change
  // object position.
  //

  glm::mat4 model = glm::translate<float>(0,0,0);//glm::mat4(1.0f);

  //
  // ModelViewProjection.
  //

  glm::mat4 MVP = projection * view * model;

  glUseProgram(programId_);

  //
  // Get location of MVP variable
  // from shader.
  //

  GLuint matrixId = glGetUniformLocation(programId_, "MVP");

  //
  // Set value to MVP.
  //

  glUniformMatrix4fv(matrixId, 1, GL_FALSE, &MVP[0][0]);

}

void GLUtil::resetPosition()
{
  mouseSensitivity_ = 1;

  zoomSpeed_ = 0;

  Fractal::setIterations(50);

  mouseX_ = 0;

  mouseY_ = 0;

  bounduary_ = BOUNDUARY;

  computeFractal_ = 1;
}

void GLUtil::zoom()
{
	if (zoomSpeed_ > 0.05 || zoomSpeed_ < -0.05)
	{
		bounduary_ -= (bounduary_ * zoomSpeed_);
		computeFractal_ = 1;
	}
}

int GLUtil::initGL(int argc, char *argv[])
{
  if (initWindow(argc, argv) != 1)
  {
    return -1;
  }

  texWidth_ = 1024;

  texHeight_ = 1024;
  
  loadShaders("../fractal_vs_3/shaders/vertex.vsh", "../fractal_vs_3/shaders/fragment.fsh");

  initObjects();

  setScene();

  return 0;
}

void GLUtil::cleanGL()
{

}

void GLUtil::startRender()
{
  glutMainLoop();
}

GLuint GLUtil::getTextureId()
{
  return textureId_ /*bufferObj*/;
}
