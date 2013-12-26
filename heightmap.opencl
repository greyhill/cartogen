#define M_PI 3.1415f

// {{{ celestial_exposure
kernel void celestial_exposure(
    int nx, int ny,
    float dx, float dy,
    float theta,
    float phi,
    global float *terrain,
    global float *luminosity) {
  int ix = get_global_id(0);
  int iy = get_global_id(1);

  float lumens = 1.f;
  float zz = terrain[ix + nx*iy];

  // compute bearing
  float hx = sin(theta) * sin(phi);
  float hy = cos(phi);
  float hz = sin(phi) * cos(theta);

  // normalize bearing vector
  float denom = fmax(fabs(hx), fabs(hy));
  hx /= denom;
  hy /= denom;
  hz /= denom;

  // tracking variables
  int ixx = ix, iyy = iy;
  float fxx = ix, fyy = iy, fzz = zz;
  while(ixx >= 0 && iyy >= 0 && ixx < nx && iyy < ny) {
    float tz = terrain[ixx + nx*iyy];
    lumens *= (zz >= tz);

    zz += hz;
    fxx += hx;
    fyy += hy;
    ixx = fxx;
    iyy = fyy;
  }

  luminosity[ix + nx*iy] = lumens;
}
// }}}

