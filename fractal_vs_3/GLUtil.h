#ifndef GLUTIL_H_
#define GLUTIL_H_
#include <Windows.h>
#include <GL/glew.h>

class GLUtil
{
  public:
  static int initGL(int argc, char *argv[]);

  static void cleanGL();

  static void startRender();

  static GLuint getTextureId();

  static int getTexWidth();

  static int getTexHeight();

  static double getZoom();
  static double getBounduary();

  static double getMouseX();
  static double getMouseY();
  static bool  getQuad();

  private:
  static void loadShaders(const char *vertexFilepath, const char *fragmentFilepath);
  static void computeFPS();
  static void displayHandler();
  static void keyboardSpecialHandler(int key, int x, int y);
  static void keyboardHandler(unsigned char key, int x, int y);
  static void mouseHandler(int button, int state, int x, int y);
  static void passiveMouseHandler(int x, int y);
  static void reshapeHandler(int x, int y);
  static void idleHandler();
  static int initWindow(int argc, char *argv[]);
  static void initObjects();
  static void setScene();
  static void resetPosition();
  static void centerAt(int x, int y);

  //
  // Zoom in, or out.
  //

  static void zoom();
};

#endif /* GLUTIL_H_ */
