#include <Fepic/CustomEigen>
#include <cmath>
#include <iostream>
using namespace Eigen;
using namespace std;

// space
typedef Matrix<double, Dynamic,1,0,6,1>              Vector;
typedef Matrix<double, Dynamic,Dynamic,RowMajor,3,3> Tensor;
typedef Matrix<double, 3, 3> Tensor3;
typedef Matrix<double, Dynamic,Dynamic,RowMajor,6,6> TensorS;
const double pi  = 3.141592653589793;
const double pi2 = pi*pi;

double pho(Vector const& X, int tag);
double gama(Vector const& X, double t, int tag);
double muu(int tag);
Vector force(Vector const& X, double t, int tag);   //density*gravity (force/vol)
Vector u_exact(Vector const& X, double t, int tag);
double p_exact(Vector const& X, double t, int tag);
Vector s_exact(int dim, double t, int tag, int LZ);
Tensor grad_u_exact(Vector const& X, double t, int tag);
Vector grad_p_exact(Vector const& X, double t, int tag);
Vector traction(Vector const& X, Vector const& normal, double t, int tag);
Vector u_initial(Vector const& X, int tag);
double p_initial(Vector const& X, int tag);
Vector s_initial(int dim, int tag, int LZ);
Vector solid_normal(Vector const& X, double t, int tag);
Tensor feature_proj(Vector const& X, double t, int tag);
Vector gravity(Vector const& X, int dim, int LZ);
Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta);
Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta);
Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta);
Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta);
Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta);
Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta);
TensorS MI_tensor(double M, double R, int dim, Tensor3 TI, int LZ);
TensorS MU_tensor(double Kelast, int dim, int LZ);
Matrix3d RotM(double theta, Matrix3d Qr, int dim);
Matrix3d RotM(double theta, int dim);
Vector SlipVel(Vector const& X, Vector const& XG, Vector const& normal,
               int dim, int tag, double theta, double Kforp, double nforp, double t,
               Matrix3d const& Q, int thetaDOF);
Vector FtauForce(Vector const& X, Vector const& XG, Vector const& normal,
                 int dim, int tag, double theta, double Kforp, double nforp, double t,
                 Vector const& Vs, Matrix3d const& Q, int thetaDOF, double Kcte);
double DFtauForce(Vector const& X, Vector const& XG, Vector const& normal,
                  int dim, int tag, double theta, double Kforp, double nforp, double t,
                  Vector const& Vs, double Kcte);
VectorXi DOFS_elimination(int LZ);

double Dif_coeff(int tag);
double nuB_coeff(int tag);
double sig_coeff(int tag);
double bbG_coeff(Vector const& X, int tag);

Vector traction_maxwell(Vector const& E, Vector const& normal, double eps, int tag);
double per_Elect(int tag);

Vector exact_tangent_ellipse(Vector const& X, Vector const& Xc, double theta, double R1, double R2, int dim);
Vector exact_normal_ellipse(Vector const& X, Vector const& Xc, double theta, double R1, double R2, int dim);

Vector cubic_ellipse(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim);
Vector Dcubic_ellipse(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim);
Vector curved_Phi(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim);
Vector Dcurved_Phi(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim);

double atan2PI(double a, double b);
Vector exact_ellipse(double yb, Vector const& X0, Vector const& X2,
                     Vector const& Xc, double theta, double R1, double R2, int dim);
Vector Dexact_ellipse(double yb, Vector const& X0, Vector const& X2,
                     Vector const& Xc, double theta, double R1, double R2, int dim);
double Flink(double t, int Nl);
double DFlink(double t, int Nl);
Vector Fdrag(int LZ);

double Ellip_arcl_integrand(double zi);
double S_arcl(double z, double zc);

double ElastPotEner(double Kelast, double xi, double R);
double DElastPotEner(double Kelast, double xi, double R);
double DDElastPotEner(double Kelast, double xi, double R);

// gota estática 2d/////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
  if (tag == 102)
  {
    return 6.4e1; //1.0e3;
  }
  else
  {
    return 1.0; //8.0e2;
  }
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 1.0;
}

double muu(int tag)
{
  if (tag == 102)
  {
    return 1.0e3;
  }
  else
  {
    return 1.0;
  }
}

Vector force(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
  //f(1) = 10;
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double un = 998*9.8*2.85e-4*2.85e-4/(3*1e-3);
  double w1 = 0.0, w2 = 1.0, R1 = .25, R2 = 1;
  //Vector v(Vector::Ones(X.size()));  v << 1 , 2;
  Vector v(Vector::Zero(X.size()));
  double r  = sqrt(x*x+y*y);
  double Fr = (R2*w2-R1*w1)*(r-R1)/(R2-R1) + R1*w1;
  double Lr = r*w2;
  double nu = muu(tag)/pho(X,tag);
  double C  = 78.0;
  double uc = exp(-C*nu*t)*(Fr-Lr)+Lr;
  v(0) = -y*Lr/r;
  v(1) =  x*Lr/r;
  if ( t == 0 ){
    if (tag == 1){
      v(0) = -w2*y;
      v(1) =  w2*x;
    }
    else if (tag == 1 && t >= 3 && false){
      v(0) =  w2*y;
      v(1) = -w2*x;
    }
    else{
      v(0) = 0;
      v(1) = 0;
    }
  }
  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double w2 = 2.0;
  Tensor dxU(Tensor::Zero(X.size(), X.size()));
  dxU(0,0) = 0; dxU(0,1) = -w2;
  dxU(1,0) = w2; dxU(1,1) = 0;

  return dxU;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0.0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  Tensor dxU(grad_u_exact(X,t,tag));
  Tensor I(Tensor::Identity(2,2));
  T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const tet = 10. * (pi/180.);
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  return f;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{
  double w2 = 2.0;
  int dim = X.size();
  int LZ = 3*(dim-1);
  Vector v(Vector::Zero(LZ)); //v << 0, 0, 10;
  if (t > 0){
    v(2) = w2;
  }
  //Vector v(Vector::Ones(LZ));
  return v;
}

Vector gravity(Vector const& X, int dim){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(3*(dim-1)));
  return f;
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(X,0,tag,LZ);
}

#endif

// solid asc 2d/////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
//  if (tag == 15)
//  {
    return 1.0;//e3;///1e4;
//  }
//  else
//  {
//    return 0.0;
//  }
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 0.5;
}

double muu(int tag)
{
//  if (tag == 15)
//  {
    return 1e-1;//1.0*0.1;
//  }
//  else
//  {
//    return 0.0;
//  }
}

Vector force(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
//  if (tag == 15)
//  {
    f(1) = -980.0*1.0;//pho(X,tag);//*1e4;//*1e3;
//
//  else
//  {
//    f(1) = 0.0;  //-8e-4*1e4;
//  }
  return f;
}

Vector gravity(Vector const& X, int dim){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(3*(dim-1)));
  if (dim == 2){
    f(1) = -980.0;//-8e-4;  //*1e3;
  }
  else if (dim == 3){
    f(2) = -980.0;  //-8e-4*1e4;
  }
  return f;
}


Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double un = 998*9.8*2.85e-4*2.85e-4/(3*1e-3);
  //Vector v(Vector::Ones(X.size()));  v << 1 , 2;
  Vector v(Vector::Zero(X.size()));

  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Tensor dxU(Tensor::Zero(X.size(), X.size()));

  return dxU;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{
  int dim = X.size();
  int LZ = 3*(dim-1);
  Vector v(Vector::Zero(LZ)); //v << .1, .2, .3;
  //Vector v(Vector::Ones(LZ));
  return v;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  
  return 0.0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  Tensor dxU(grad_u_exact(X,t,tag));
  Tensor I(Tensor::Identity(2,2));
  T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(X,0,tag,LZ);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  return f;
}

Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (dij <= Ri+Rj){
    f = (1/ep1)*(Ri+Rj-dij)*(Xi - Xj);
  }
  else if((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f = (1/ep2)*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj);
  }
  return f;
}

Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double di = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (di <= 2*Ri){
    f = (1/ew1)*(2*Ri-di)*(Xi - Xj);
  }
  else if((2*Ri <= di) && (di <= 2*Ri+zeta)){
    f = (1/ew2)*(2*Ri+zeta-di)*(2*Ri+zeta-di)*(Xi - Xj);
  }
  return f;
}

Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  f = (zeta/ep)*(Xi - Xj)/(Xi - Xj).norm();
  return f;
}

Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  double g = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    f   = masj*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = dij = (Xi - Xj).norm();
  double g = 0.0;
  double R = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    R   = std::max(Ri,Rj);
    f   = (rhoj-rhof)*pi*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f   = (Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/ep;
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}
#endif

// canal /////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
  return 1.0;
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0*5.e-1;
}

double beta_diss()
{
  return 0*1.e-4;
}

double gama(Vector const& X, double t, int tag)
{
  return 0;
}

double muu(int tag)
{
  return  1.0;
}

Vector force(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double umax = 1;
  Vector v(Vector::Zero(X.size()));

  //if (tag == 3 || tag == 2)
  if (tag == 2)
  {
    v(0) = umax*y*(2-y);
    v(1) = 0;
  }

  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Tensor dxU(Tensor::Zero(X.size(), X.size()));

  return dxU;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  Tensor dxU(grad_u_exact(X,t,tag));
  Tensor I(Tensor::Identity(2,2));
  T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const tet = 10. * (pi/180.);
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //N(1) = 1;
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  if (tag == 3)
  {
    f(0,0) = 1;
  }
//  else if (tag == 1)
//  {
//    f(0,0) = 1;
//  }
  return f;
}

#endif

// rot solid 2d/////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
//  if (tag == 15)
//  {
    return 1.0;//e3;///1e4;
//  }
//  else
//  {
//    return 0.0;
//  }
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 0.5;
}

double muu(int tag)
{
//  if (tag == 15)
//  {
    return 1e-2;//1.0*0.1;
//  }
//  else
//  {
//    return 0.0;
//  }
}

Vector force(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
//  if (tag == 15)
//  {
//    f(1) = -980.0*1.0;//pho(X,tag);//*1e4;//*1e3;
//
//  else
//  {
//    f(1) = 0.0;  //-8e-4*1e4;
//  }
  return f;
}

Vector gravity(Vector const& X, int dim){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(3*(dim-1)));
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double un = 998*9.8*2.85e-4*2.85e-4/(3*1e-3);
  double w1 = 0.0, w2 = 2.0, R1 = .125, R2 = .5;
  //Vector v(Vector::Ones(X.size()));  v << 1 , 2;
  Vector v(Vector::Zero(X.size()));
  double r  = sqrt(x*x+y*y);
  double Fr = (R2*w2-R1*w1)*(r-R1)/(R2-R1) + R1*w1;
  double Lr = r*w2;
  double nu = muu(tag)/pho(X,tag);
  double C  = 78.0;
  double uc = exp(-C*nu*t)*(Fr-Lr)+Lr;
  v(0) = -y*Lr/r;
  v(1) =  x*Lr/r;
  if ( t == 0 ){
    if (tag == 1){
      v(0) = -w2*y;
      v(1) =  w2*x;
    }
    else if (tag == 1 && t >= 3 && false){
      v(0) =  w2*y;
      v(1) = -w2*x;
    }
    else{
      v(0) = 0;
      v(1) = 0;
    }
  }
  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double w2 = 2.0;
  Tensor dxU(Tensor::Zero(X.size(), X.size()));
  dxU(0,0) = 0; dxU(0,1) = -w2;
  dxU(1,0) = w2; dxU(1,1) = 0;

  return dxU;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{
  double w2 = 2.0;
  int dim = X.size();
  int LZ = 3*(dim-1);
  Vector v(Vector::Zero(LZ)); //v << 0, 0, 10;
  if (t > 0){
    v(2) = w2;
  }
  //Vector v(Vector::Ones(LZ));
  return v;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0.0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  Tensor dxU(grad_u_exact(X,t,tag));
  Tensor I(Tensor::Identity(2,2));
  T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(X,0,tag,LZ);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  return f;
}

Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (dij <= Ri+Rj){
    f = (1/ep1)*(Ri+Rj-dij)*(Xi - Xj);
  }
  else if((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f = (1/ep2)*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj);
  }
  return f;
}

Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double di = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (di <= 2*Ri){
    f = (1/ew1)*(2*Ri-di)*(Xi - Xj);
  }
  else if((2*Ri <= di) && (di <= 2*Ri+zeta)){
    f = (1/ew2)*(2*Ri+zeta-di)*(2*Ri+zeta-di)*(Xi - Xj);
  }
  return f;
}

Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  f = (zeta/ep)*(Xi - Xj)/(Xi - Xj).norm();
  return f;
}

Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  double g = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    f   = masj*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = dij = (Xi - Xj).norm();
  double g = 0.0;
  double R = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    R   = std::max(Ri,Rj);
    f   = (rhoj-rhof)*pi*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f   = (Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/ep;
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}
/*
TensorS MI_tensor(double M, double R, int dim)
{
  TensorS MI(TensorS::Zero(3*(dim-1),3*(dim-1)));
  if (dim == 2){
    MI(0,0) = M; MI(1,1) = M; MI(2,2) = 0.5*M*R*R;
  }
  else if(dim == 3){
    MI(0,0) = M; MI(1,1) = M; MI(2,2) = M;
    MI(3,3) = 0.4*M*R*R; MI(4,4) = 0.4*M*R*R; MI(5,5) = 0.4*M*R*R;
  }
  return MI;
}

Vector SlipVel(Vector const& X, Vector const& XG, int dim, int tag)
{
  Vector V(Vector::Zero(dim));
  Vector X3(Vector::Zero(3));
  Vector Xp(Vector::Zero(dim));
  double alp = 0.1;
  double bet = 0.1;

  if (dim == 2)
  {
    X3(0) = X(0); X3(1) = X(1);
    X3 = X3 - XG;
    X3.normalize();
    Xp(0) = X3(1); Xp(1) = -X3(0);
    if (tag == 104){
      V = alp*Xp;
    }
    else if (tag == 105){
      V = -bet*Xp;
    }
  }

  return V;
}
*/
#endif

// rot solid 2d slip vel/////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
//  if (tag == 15)
//  {
    return 1.0;//e3;///1e4;
//  }
//  else
//  {
//    return 0.0;
//  }
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 0.5;
}

double muu(int tag)
{
//  if (tag == 15)
//  {
    return 1;//1.0*0.1;
//  }
//  else
//  {
//    return 0.0;
//  }
}

Vector force(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
//  if (tag == 15)
//  {
//    f(1) = -980.0*1.0;//pho(X,tag);//*1e4;//*1e3;
//
//  else
//  {
//    f(1) = 0.0;  //-8e-4*1e4;
//  }
  return f;
}

Vector gravity(Vector const& X, int dim){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(3*(dim-1)));
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double un = 998*9.8*2.85e-4*2.85e-4/(3*1e-3);
  double w1 = 0.0, w2 = 1.0, R1 = .25, R2 = 1;
  //Vector v(Vector::Ones(X.size()));  v << 1 , 2;
  Vector v(Vector::Zero(X.size()));
/*  double r  = sqrt(x*x+y*y);
  double Fr = (R2*w2-R1*w1)*(r-R1)/(R2-R1) + R1*w1;
  double Lr = r*w2;
  double nu = muu(tag)/pho(X,tag);
  double C  = 78.0;
  double uc = exp(-C*nu*t)*(Fr-Lr)+Lr;
  v(0) = -y*Lr/r;
  v(1) =  x*Lr/r;
  if ( t == 0 ){
    if (tag == 1){
      v(0) = -w2*y;
      v(1) =  w2*x;
    }
    else if (tag == 1 && t >= 3 && false){
      v(0) =  w2*y;
      v(1) = -w2*x;
    }
    else{
      v(0) = 0;
      v(1) = 0;
    }
  }*/
  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double w2 = 2.0;
  Tensor dxU(Tensor::Zero(X.size(), X.size()));
  dxU(0,0) = 0; dxU(0,1) = -w2;
  dxU(1,0) = w2; dxU(1,1) = 0;

  return dxU;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{
  double w2 = 2.0;
  int dim = X.size();
  int LZ = 3*(dim-1);
  Vector v(Vector::Zero(LZ)); //v << 0, 0, 10;
  if (t > 0){
    v(2) = w2;
  }
  //Vector v(Vector::Ones(LZ));
  return v;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0.0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  Tensor dxU(grad_u_exact(X,t,tag));
  Tensor I(Tensor::Identity(2,2));
  T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(X,0,tag,LZ);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  return f;
}

Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (dij <= Ri+Rj){
    f = (1/ep1)*(Ri+Rj-dij)*(Xi - Xj);
  }
  else if((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f = (1/ep2)*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj);
  }
  return f;
}

Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double di = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (di <= 2*Ri){
    f = (1/ew1)*(2*Ri-di)*(Xi - Xj);
  }
  else if((2*Ri <= di) && (di <= 2*Ri+zeta)){
    f = (1/ew2)*(2*Ri+zeta-di)*(2*Ri+zeta-di)*(Xi - Xj);
  }
  return f;
}

Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  f = (zeta/ep)*(Xi - Xj)/(Xi - Xj).norm();
  return f;
}

Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  double g = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    f   = masj*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = dij = (Xi - Xj).norm();
  double g = 0.0;
  double R = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    R   = std::max(Ri,Rj);
    f   = (rhoj-rhof)*pi*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f   = (Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/ep;
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

#endif

// rot solid 3d slip vel/////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
//  if (tag == 15)
//  {
    return 1.0;//e3;///1e4;
//  }
//  else
//  {
//    return 0.0;
//  }
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 0.5;
}

double muu(int tag)
{
//  if (tag == 15)
//  {
    return 0.1;//1.0*0.1;
//  }
//  else
//  {
//    return 0.0;
//  }
}

Vector force(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
//  if (tag == 15)
//  {
//    f(1) = -980.0*1.0;//pho(X,tag);//*1e4;//*1e3;
//
//  else
//  {
//    f(1) = 0.0;  //-8e-4*1e4;
//  }
  return f;
}

Vector gravity(Vector const& X, int dim){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(3*(dim-1)));
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double un = 998*9.8*2.85e-4*2.85e-4/(3*1e-3);
  double w1 = 0.0, w2 = 1.0, R1 = .25, R2 = 1;
  //Vector v(Vector::Ones(X.size()));  v << 1 , 2;
  Vector v(Vector::Zero(X.size()));
  if (tag == 101 || tag == 102 || tag == 103){
    v(0) = 0.333328;
  }
/*  double r  = sqrt(x*x+y*y);
  double Fr = (R2*w2-R1*w1)*(r-R1)/(R2-R1) + R1*w1;
  double Lr = r*w2;
  double nu = muu(tag)/pho(X,tag);
  double C  = 78.0;
  double uc = exp(-C*nu*t)*(Fr-Lr)+Lr;
  v(0) = -y*Lr/r;
  v(1) =  x*Lr/r;
  if ( t == 0 ){
    if (tag == 1){
      v(0) = -w2*y;
      v(1) =  w2*x;
    }
    else if (tag == 1 && t >= 3 && false){
      v(0) =  w2*y;
      v(1) = -w2*x;
    }
    else{
      v(0) = 0;
      v(1) = 0;
    }
  }*/
  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double w2 = 2.0;
  Tensor dxU(Tensor::Zero(X.size(), X.size()));
  dxU(0,0) = 0; dxU(0,1) = -w2;
  dxU(1,0) = w2; dxU(1,1) = 0;

  return dxU;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{
  double w2 = 2.0;
  int dim = X.size();
  int LZ = 3*(dim-1);
  Vector v(Vector::Zero(LZ)); v << 0.333328, 0, 0, 0, 0, 0;
  //Vector v(Vector::Ones(LZ));
  return v;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0.0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  Tensor dxU(grad_u_exact(X,t,tag));
  Tensor I(Tensor::Identity(2,2));
  T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(X,0,tag,LZ);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  return f;
}

Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (dij <= Ri+Rj){
    f = (1/ep1)*(Ri+Rj-dij)*(Xi - Xj);
  }
  else if((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f = (1/ep2)*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj);
  }
  return f;
}

Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double di = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (di <= 2*Ri){
    f = (1/ew1)*(2*Ri-di)*(Xi - Xj);
  }
  else if((2*Ri <= di) && (di <= 2*Ri+zeta)){
    f = (1/ew2)*(2*Ri+zeta-di)*(2*Ri+zeta-di)*(Xi - Xj);
  }
  return f;
}

Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  f = (zeta/ep)*(Xi - Xj)/(Xi - Xj).norm();
  return f;
}

Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  double g = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    f   = masj*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = dij = (Xi - Xj).norm();
  double g = 0.0;
  double R = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    R   = std::max(Ri,Rj);
    f   = (rhoj-rhof)*pi*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f   = (Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/ep;
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

#endif

// rot solid 2d trap and channel 2018/////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
//  if (tag == 15)
//  {
    return 1000;//e3;///1e4;
//  }
//  else
//  {
//    return 0.0;
//  }
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 0.5;
}

double muu(int tag)
{
//  if (tag == 15)
//  {
    return 1e-3;//1.0*0.1;
//  }
//  else
//  {
//    return 0.0;
//  }
}

Vector force(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
//  if (tag == 15)
//  {
//    f(1) = -980.0*1.0;//pho(X,tag);//*1e4;//*1e3;
//
//  else
//  {
//    f(1) = 0.0;  //-8e-4*1e4;
//  }
  return f;
}

Vector gravity(Vector const& X, int dim){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(3*(dim-1)));
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector v(Vector::Zero(X.size()));
  double Um = 10/*e-6*/, H = 200/*e-6*/;
  if (tag == 4){
    //Um = Um*(1-exp(-t*t*t*t*1e10));//Um*(1-exp(t*t*t*t/1e-14));//Um*(-1-exp(t*t*t*t/1e-14)*0);
    v(0) = Um*4*(H-y)*y/(H*H);
  }
  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double w2 = 2.0, Um = 0.1, H = 2e-6;
  Tensor dxU(Tensor::Zero(X.size(), X.size()));
  dxU(0,0) = 0; dxU(0,1) = Um/H - Um*2.0*y/H;
  dxU(1,0) = 0; dxU(1,1) = 0;

  return dxU;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{
  double w2 = 0.0;
  Vector v(Vector::Zero(LZ)); //v << 0, 0, 10;
  if (t > 0){
    v(0) = 0.1;
  }
  //Vector v(Vector::Ones(LZ));
  return v;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0.0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  Tensor dxU(grad_u_exact(X,t,tag));
  Tensor I(Tensor::Identity(2,2));
  T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(dim,0,tag,LZ);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  if (tag == 3){
    f(0,0) = 1;
  }
  return f;
}

Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (dij <= Ri+Rj){
    f = (1/ep1)*(Ri+Rj-dij)*(Xi - Xj);
  }
  else if((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f = (1/ep2)*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj);
  }
  return f;
}

Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double di = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (di <= 2*Ri){
    f = (1/ew1)*(2*Ri-di)*(Xi - Xj);
  }
  else if((2*Ri <= di) && (di <= 2*Ri+zeta)){
    f = (1/ew2)*(2*Ri+zeta-di)*(2*Ri+zeta-di)*(Xi - Xj);
  }
  return f;
}

Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  f = (zeta/ep)*(Xi - Xj)/(Xi - Xj).norm();
  return f;
}

Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  double g = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    f   = masj*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = dij = (Xi - Xj).norm();
  double g = 0.0;
  double R = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    R   = std::max(Ri,Rj);
    f   = (rhoj-rhof)*pi*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f   = (Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/ep;
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

#endif

// rot solid 2d axis: Reynolds calculations/////////////////////////////////////////////////////////////
#if (false)

double pho(Vector const& X, int tag)
{
//  if (tag == 15)
//  {
//    double alp = 100.0, bet = 0.018, kst = (double)tag, alp1 = 8.0;
//    if (kst <= alp1){
//      return pow(10.0,(kst+alp1)/alp1);//e3;///1e4;
//    }
//    else{
//      return pow(10.0,2.0 + bet*(kst-alp1) + (1.0/(alp*alp) - bet/alp)*(kst-alp1)*(kst-alp1));
//    }
/*  double kst = (double)tag, p1 = -3, alp1 = 10.0, alp2 = 20.0, alp3 = 30.0, alp4 = 40.0, alp5 = 50.0;
  if (kst <= alp1)
    return pow(10,p1+0) + (kst     )*(pow(10,p1+1) - pow(10,p1+0))/(alp1     );
  else if (kst > alp1 && kst <= alp2)
    return pow(10,p1+1) + (kst-alp1)*(pow(10,p1+2) - pow(10,p1+1))/(alp2-alp1);
  else if (kst > alp2 && kst <= alp3)
    return pow(10,p1+2) + (kst-alp2)*(pow(10,p1+3) - pow(10,p1+2))/(alp3-alp2);
  else if (kst > alp3 && kst <= alp4)
    return pow(10,p1+3) + (kst-alp3)*(pow(10,p1+4) - pow(10,p1+3))/(alp4-alp3);
  else if (kst > alp4 && kst <= alp5)
    return pow(10,p1+4) + (kst-alp4)*(pow(10,p1+5) - pow(10,p1+4))/(alp5-alp4);
  else
    return -1000000;*/
  return 0*1.0e-12; /* gr/um^3 */; //0.0;
//  }
//  else
//  {
//    return 0.0;
//  }
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 0.5;
}

double muu(int tag)
{
//  if (tag == 15)
//  {
    return 100;//1.0e-6;// 1.0e-6 /* gr/(um sec) */; //1.0e-3;//1.0/3.0;//1.0*0.1;
//  }
//  else
//  {
//    return 0.0;
//  }
}

Vector force(Vector const& X, double t, int tag)//gravity*pho
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
//  if (tag == 15)
//  {
    f(1) = 0; //-1;//-980.0*1.0;//pho(X,tag);//*1e4;//*1e3;
//
//  else
//  {
//    f(1) = 0.0;  //-8e-4*1e4;
//  }
  return f;
}

Vector gravity(Vector const& X, int dim, int LZ){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(LZ));
  if (dim == 2){
    f(1) = 0*(+1.0);//-1.0e-3; //-1;//-980.0;//-8e-4;  //*1e3;
  }
  else if (dim == 3){
    f(2) = -980.0;  //-8e-4*1e4;
  }
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector v(Vector::Zero(X.size()));
  double Um = 1/*e-6*/, H = 5;//e-6;
  if ( false && (tag == 3 || tag == 2 || tag == 6 || tag == 7) ){
    //Um = Um*(1-exp(-t*t*t*t*1e10));//Um*(1-exp(t*t*t*t/1e-14));//Um*(-1-exp(t*t*t*t/1e-14)*0);
    //v(0) = Um*4*(H-y)*y/(H*H); v(1) = 0.0;
    v(1) = Um*(H-x)*(H+x)/(H*H); v(0) = 0.0;
  }
  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double w2 = 2.0, Um = 0.1, H = 2e-6;
  Tensor dxU(Tensor::Zero(X.size(), X.size()));
  //dxU(0,0) = 0; dxU(0,1) = Um/H - Um*2.0*y/H;
  //dxU(1,0) = 0; dxU(1,1) = 0;

  return dxU;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{//for inertialess case, this MUST be zero at t = 0;
  double w2 = 0.0;
  Vector v(Vector::Zero(LZ)); //v << 1, 1, 0;
  //Vector v(Vector::Ones(LZ));
  return v;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0.0;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  //Tensor dxU(grad_u_exact(X,t,tag));
  //Tensor I(Tensor::Identity(2,2));
  //T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(dim,0,tag,LZ);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));
  v(1) = 1.0;

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  //if (true && (tag == 1 || tag == 5 || tag == 2 || tag == 4 || tag == 6 /*|| tag == 3 || tag == 1*/)){
    //f(0,0) = 1; //imposes zero tangential velocity at the output of the channel
                //or cartesian wall by eliminating the contribution of the momentum
                //equation in the normal direction, allowing penetration with zero
                //stress in the normal direction. In other words,
                //indicates that the component 0 (x-direction) is free, and is going
                //to be imposed zero velocity in the component 1 (y-direction)
    //f(1,1) = 1;
  //}
  //else if (true && (tag == 3 || tag == 7)){
  //if (true && (tag == 4)){
    //f(1,1) = 1;
  //}
  //if (tag == 4 || tag == 5 || tag == 8){f(1,1) = 1;}
  //if (tag == 6 || tag == 5 || tag == 1 || tag == 8){
    f(1,1) = 1.0;  //1
  //}
  return f;
}

Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (dij <= Ri+Rj){
    f = (1/ep1)*(Ri+Rj-dij)*(Xi - Xj);
  }
  else if((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f = (1/ep2)*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj);
  }
  return f;
}

Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double di = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (di <= 2*Ri){
    f = (1/ew1)*(2*Ri-di)*(Xi - Xj);
  }
  else if((2*Ri <= di) && (di <= 2*Ri+zeta)){
    f = (1/ew2)*(2*Ri+zeta-di)*(2*Ri+zeta-di)*(Xi - Xj);
  }
  return f;
}

Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  f = (zeta/ep)*(Xi - Xj)/(Xi - Xj).norm();
  return f;
}

Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  double g = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    f   = masj*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = dij = (Xi - Xj).norm();
  double g = 0.0;
  double R = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    R   = std::max(Ri,Rj);
    f   = (rhoj-rhof)*pi*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f   = (Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/ep;
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

#endif

// junction 2018/////////////////////////////////////////////////////////////
#if (true)

double pho(Vector const& X, int tag)
{
  return 1000.0;
}

double cos_theta0()
{
  return 0.0;
}

double zeta(double u_norm, double angle)
{
  return 0.0;
}

double beta_diss()
{
  return 0.0;
}

double gama(Vector const& X, double t, int tag)
{
  return 0.0;
}

double muu(int tag)
{
    return 1e-3;
}

Vector force(Vector const& X, double t, int tag)//gravity*pho
{
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(X.size()));
  f(1) = 0;
  return f;
}

Vector gravity(Vector const& X, int dim, int LZ){
  double x = X(0);
  double y = X(1);

  Vector f(Vector::Zero(LZ));
  if (dim == 2){
    f(1) = 0.0;//-1.0e-3; //-1;//-980.0;//-8e-4;  //*1e3;
  }
  else if (dim == 3){
    f(2) = -980.0;  //-8e-4*1e4;
  }
  return f;
}

Vector u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector v(Vector::Zero(X.size()));
  double Um = 1e-6, H = 100e-6;
  if ( true && (tag == 2) ){
    //Um = Um*(1-exp(-t*t*t*t*1e10));//Um*(1-exp(t*t*t*t/1e-14));//Um*(-1-exp(t*t*t*t/1e-14)*0);
    //v(0) = Um*4*(H-y)*y/(H*H); v(1) = 0.0;
    v(0) = Um*(H-y)*(H+y)/(H*H);
  }
  if ( true && (tag == 3) ){
    //Um = Um*(1-exp(-t*t*t*t*1e10));//Um*(1-exp(t*t*t*t/1e-14));//Um*(-1-exp(t*t*t*t/1e-14)*0);
    //v(0) = Um*4*(H-y)*y/(H*H); v(1) = 0.0;
    v(0) = -Um*(H-y)*(H+y)/(H*H);
  }
  return v;
}

Tensor grad_u_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  double w2 = 2.0, Um = 0.1, H = 2e-6;
  Tensor dxU(Tensor::Zero(X.size(), X.size()));
  //dxU(0,0) = 0; dxU(0,1) = Um/H - Um*2.0*y/H;
  //dxU(1,0) = 0; dxU(1,1) = 0;

  return dxU;
}

Vector s_exact(int dim, double t, int tag, int LZ)
{//for inertialess case, this MUST be zero at t = 0;
  double w2 = 0.0;
  Vector v(Vector::Zero(LZ)); //v << 1, 1, 0;
  //Vector v(Vector::Ones(LZ));
  return v;
}

double p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);

  return 0.0; //3.5;
}

Vector grad_p_exact(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector dxP(X.size());

  return dxP;
}

Vector traction(Vector const& X, Vector const& normal, double t, int tag)
{
  Vector T(Vector::Zero(X.size()));
  //T(0) = -p_exact(X,t,tag);
  //T(1) = muu(tag)*(cos(w_*t) + sin(w_*t));
  //Tensor dxU(grad_u_exact(X,t,tag));
  //Tensor I(Tensor::Identity(2,2));
  //T = (- p_exact(X,t,tag)*I +  muu(tag)*(dxU + dxU.transpose()))*normal;
  return T;
}

Vector u_initial(Vector const& X, int tag)
{
  return u_exact(X,0,tag);
}

Vector s_initial(int dim, int tag, int LZ)
{
  return s_exact(dim,0,tag,LZ);
}

double p_initial(Vector const& X, int tag)
{
  return p_exact(X,0,tag);
}

Vector solid_normal(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  return N;
}

Vector v_exact(Vector const& X, double t, int tag) //(X,t,tag)
{
  double const x = X(0);
  double const y = X(1);
  Vector v(Vector::Zero(X.size()));
  //v(1) = 1.0;

  return v;
}

// posição do contorno
Vector x_exact(Vector const& X, double t, int tag)
{
  Vector r(Vector::Zero(X.size()));
  return r;
}

Vector solid_veloc(Vector const& X, double t, int tag)
{
  Vector N(Vector::Zero(X.size()));
  //if (tag == 2) {N(1) = 1;};
  return N;
}

Tensor feature_proj(Vector const& X, double t, int tag)
{
  Tensor f(Tensor::Zero(X.size(), X.size()));
  //if (true && (tag == 1 || tag == 5 || tag == 2 || tag == 4 || tag == 6 /*|| tag == 3 || tag == 1*/)){
    //f(0,0) = 1; //imposes zero tangential velocity at the output of the channel
                //or cartesian wall by eliminating the contribution of the momentum
                //equation in the normal direction, allowing penetration with zero
                //stress in the normal direction. In other words,
                //indicates that the component 0 (x-direction) is free, and is going
                //to be imposed zero velocity in the component 1 (y-direction)
    //f(1,1) = 1;
  //}
  //else if (true && (tag == 3 || tag == 7)){
  if (true && (tag == 4)){
    f(1,1) = 1;
  }
  if (true && (tag == 5)){
    f(1,1) = 1;
  }
  return f;
}

Vector force_pp(Vector const& Xi, Vector const& Xj, double Ri, double Rj,
                 double ep1, double ep2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (dij <= Ri+Rj){
    f = (1/ep1)*(Ri+Rj-dij)*(Xi - Xj);
  }
  else if((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f = (1/ep2)*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj);
  }
  return f;
}

Vector force_pw(Vector const& Xi, Vector const& Xj, double Ri,
                 double ew1, double ew2, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double di = (Xi - Xj).norm();
//  if (dij > Ri+Rj+zeta){
//    return f;
//  }
  if (di <= 2*Ri){
    f = (1/ew1)*(2*Ri-di)*(Xi - Xj);
  }
  else if((2*Ri <= di) && (di <= 2*Ri+zeta)){
    f = (1/ew2)*(2*Ri+zeta-di)*(2*Ri+zeta-di)*(Xi - Xj);
  }
  return f;
}

Vector force_ppl(Vector const& Xi, Vector const& Xj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  f = (zeta/ep)*(Xi - Xj)/(Xi - Xj).norm();
  return f;
}

Vector force_rga(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const masj, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  double g = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    f   = masj*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgb(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 Vector const& Gr, double const rhoj, double const rhof, double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = dij = (Xi - Xj).norm();
  double g = 0.0;
  double R = 0.0;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    g   = Gr.norm();
    R   = std::max(Ri,Rj);
    f   = (rhoj-rhof)*pi*g*(Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/(ep*zeta*zeta*dij);
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

Vector force_rgc(Vector const& Xi, Vector const& Xj, double const Ri, double const Rj,
                 double ep, double zeta)
{
  Vector f(Vector::Zero(Xi.size()));
  double dij = (Xi - Xj).norm();;
  if ((Ri+Rj <= dij) && (dij <= Ri+Rj+zeta)){
    f   = (Ri+Rj+zeta-dij)*(Ri+Rj+zeta-dij)*(Xi - Xj)/ep;
  }
  //else if (dij < Ri+Rj){cout << "ERROR: penetration!!!!!!!!!!!!!!!!" << endl;}
  return f;
}

#endif

// General functions/////////////////////////////////////////////////////////////
#if (true)
TensorS MI_tensor(double M, double R, int dim, Tensor3 TI, int LZ)
{
  TensorS MI(TensorS::Zero(LZ,LZ));
  if (dim == 2){
    MI(0,0) = M; MI(1,1) = M; MI(2,2) = TI(2,2); //MI(2,2) = 0.5*M*R*R;
  }
  else if (dim == 3){
    MI(0,0) = M; MI(1,1) = M; MI(2,2) = M;
    MI.block(3,3,5,5) = TI;
    //MI(3,3) = 0.4*M*R*R; MI(4,4) = 0.4*M*R*R; MI(5,5) = 0.4*M*R*R;
  }
  return MI;
}

TensorS MU_tensor(double Kelast, int dim, int LZ)
{
  TensorS MU(TensorS::Zero(LZ,LZ));
  if (dim == 2){
    MU(2,2) = Kelast; //MI(2,2) = 0.5*M*R*R;
  }
  else if (dim == 3){
    //MI(0,0) = M; MI(1,1) = M; MI(2,2) = M;
    //MI.block(3,3,5,5) = TI;
    //MI(3,3) = 0.4*M*R*R; MI(4,4) = 0.4*M*R*R; MI(5,5) = 0.4*M*R*R;
  }
  return MU;
}

Matrix3d RotM(double theta, Matrix3d Qr, int thetaDOF)
{
  Matrix3d M(Matrix3d::Identity(3,3)); //Matrix3d M(Matrix3d::Zero(3,3));
  if (thetaDOF == 1){
    M(0,0) = cos(theta); M(0,1) = -sin(theta);
    M(1,0) = sin(theta); M(1,1) =  cos(theta);
  }
  else {
    M = Qr;
  }
  return M;
}

Matrix3d RotM(double theta, int dim)
{
  Matrix3d M(Matrix3d::Zero(3,3));
  if (dim == 2){
    M(0,0) = cos(theta); M(0,1) = -sin(theta);
    M(1,0) = sin(theta); M(1,1) =  cos(theta);
  }
  return M;
}

Vector SlipVel(Vector const& X, Vector const& XG, Vector const& normal,
               int dim, int tag, double theta, double Kforp, double nforp, double t,
               Matrix3d const& Q, int thetaDOF)
{
  int cas = 100;
  Vector V(Vector::Zero(dim));
  Vector X3(Vector::Zero(3));
  Vector Xp(Vector::Zero(dim));

  double alp = 50.0;
  double bet = 1.0;
  Vector tau(dim);
  tau(0) = +normal(1); tau(1) = -normal(0);  //this tangent goes in the direction of the parametrization
  //tau = -tau;;//this tangent goes against the parametrization
  Tensor I(dim,dim);
  I.setIdentity();
  Tensor Pr = I - normal*normal.transpose();

  if (cas == 0 && dim == 3)
  {
    if (false && tag == 100){
      V(0) = -0.01; V(1) = -0.01;
    }
    else if (tag == 101 /*|| tag == 203*/){
      V(2) = -1.0*0.5;
    }
    V = Pr*V;  //V.normalize();
  }
  else if (cas == 1 && dim == 2) //this gives structure node swim
  {
    if (tag == 101 || tag == 102){//(tag == 103 || tag == 104)
      theta = pi;//pi/4; //5 grados appr
      V(0) = cos(theta); V(1) = sin(theta);
      V = alp*Pr*V;
    }
  }
  else if (cas == 2 && dim == 2) //this gives total random swim node
  {
    double psi = 0.0;
    for (int I = 0; I < 20; I++)
    {
      if (tag == 121 + I)
      {
        psi = pi;//rand() % (6);         // v1 in the range 0 to 99
        V(0) = cos(psi); V(1) = sin(psi);
        V = alp*Pr*V;
      }
    }
  }
  else if (cas == 3 && dim == 2)
  {//don't forget to put viscosity 1 for the unitary circular, spherical toy case
    double psi = 0.0;  theta = pi/2.0; //use this last theta when the theta_0 of the init cond has to be set as zero
    psi = atan2PI(X(1)-XG(1),X(0)-XG(0));
    double B1 = +1.0, B2 = +0.5;//B1 = +0.5, B2 = +3.0;
    double uthe = B1*sin(theta-psi) + B2*sin(theta-psi)*cos(theta-psi);
    //V(0) = +normal(1); V(1) = -normal(0);   //V(0) = -normal(1); V(1) = +normal(0);
    V = uthe*tau; V = uthe*tau;
    //if ( X(1)-XG(1) < 0.0 )
    //  V = Vector::Zero(dim);
    //else
    //  V = V*(X(1)-XG(1))/0.2;
  }
  else if (cas == 4 && dim == 2)
  {
    double psi = 0.0;
    double B1 = 0.0, B2 = 0.0;
    double omega = 1.0, k = 1.0, phi = pi/2.0, a0 = 1.0, b0 = 1.0;
    psi = atan2PI(X(1)-XG(1),X(0)-XG(0));
    B1 = 0.5*omega*k*(b0*b0 + 2*a0*b0*cos(theta-psi) - a0*a0);
    double uthe = B1*sin(theta-psi) + B2*sin(theta-psi)*cos(theta-psi);
    V(0) = +normal(1); V(1) = -normal(0);   //V(0) = -normal(1); V(1) = +normal(0);
    V = 0*uthe*V;
  }
  else if (cas == 5 && dim == 2)//for metachronal waves
  {
    Matrix3d Qr(Matrix3d::Zero(3,3));
    Vector3d Xref(Vector3d::Zero(3)), X3(Vector::Zero(3));
    X3(0) = X(0); X3(1) = X(1);
    Xref = RotM(theta,Qr,dim).transpose()*(X3 - XG);
    //cout << Xref.transpose() << "  " << X3.transpose() << "  " << XG.transpose() << endl;
    double uthe = 0.0;
    //double a = 110.0e-0, Tp = 1.0, omega = 2*pi/Tp, eta = 10;
    double a = 110.0e-0, Tp = 0.2, omega = +2*pi/Tp, eta = 10;
    double W = S_arcl(X(1), XG(1)), S;
    double L = S_arcl(-a,0.0);/*S_arcl(XG(1)-a,XG(1));*/  //cout << "for " << X(1) << "  " << W << "  " << L << endl;
    double K = Kforp*L;  //0.015*L;
    double n = nforp, k = 2*pi/L * n;
    //uthe = tanh(eta*sin(pi*S/L)) * A*omega*sin(k*S - omega*t);

    S = W;
    for (int ni = 0; ni < 200; ni++){
      double F0 = S + K*tanh(eta*sin(pi*S/L))*cos(k*S-omega*t) - W;
      double F1 = 1 + (pi*K*eta/L)*(1 - pow(tanh(eta*sin(pi*S/L)),2.0))*cos(pi*S/L)*cos(k*S-omega*t)
                  - k*K*tanh(eta*sin(pi*S/L))*sin(k*S-omega*t);
      S = S - F0/F1;
    }

    uthe = K*tanh(eta*sin(pi*S/L))*omega*sin(k*S - omega*t);
    //V(0) = +normal(1); V(1) = -normal(0);  //this tangent goes in the direction of the parametrization
    //V(0) = -normal(1); V(1) = +normal(0);//this tangent goes against the parametrization
    V = uthe*tau;
    //cout << V.transpose() << endl;
  }
  else if (cas == 6 && dim == 2 && (tag == 101 /*|| tag == 102*//*144 || tag == 142 || tag == 130*/))
  {//for metachronal waves full body
    Matrix3d Qr = Q;//(Matrix3d::Zero(3,3));
    Vector3d Xref(Vector3d::Zero(3)), X3(Vector::Zero(3));
    X3(0) = X(0); X3(1) = X(1);
    Xref = RotM(theta,Qr,thetaDOF).transpose()*(X3 - XG);
    //cout << Xref.transpose() << "  " << X3.transpose() << "  " << XG.transpose() << endl;
    double uthe = 0.0;
    //double a = 2.0e-0, Tp = 0.05, omega = -2*pi/Tp, eta = 10;
    double a = 110.0e-0, Tp = 0.2, omega = -2*pi/Tp, eta = 5;
    //double a = 110.0e-0, Tp = 1.0/40.0, omega = -2*pi/Tp, eta = 4.0;
    double W = S_arcl(Xref(0), 0.0), S;  //cout << S_arcl(Xref(0), 0.0) << "  " << S_arcl(a, 0.0) << "  " << S_arcl(-a, 0.0) << endl;
    double L = S_arcl(-a,0.0);  //cout << "for " << Xref(0) << "  " << W << "  " << L << endl;
    double K = Kforp*L;  //cout << L << endl;//0.015*L;
    double n = nforp, k = 2*pi/L * n;

    S = W;
    for (int ni = 0; ni < 100; ni++){
      double F0 = S + K*tanh(eta*sin(pi*S/L))*cos(k*S-omega*t) - W;
      double F1 = 1 + (pi*K*eta/L)*(1 - pow(tanh(eta*sin(pi*S/L)),2.0))*cos(pi*S/L)*cos(k*S-omega*t)
                  - k*K*tanh(eta*sin(pi*S/L))*sin(k*S-omega*t);
      S = S - F0/F1;
    }

    if (Xref(1) <= 0){
      uthe = K*tanh(eta*sin(pi*S/L))*omega*sin(k*S - omega*t);
    }
    else{
      uthe = -K*tanh(eta*sin(pi*S/L))*omega*sin(k*S - omega*t);
    }
    V = uthe*tau;  //cout << V.transpose() << endl;
    //if (Xref(1) <= 0 && t == 0.0){
    //  cout << X(0) << " ";
    //}
  }
  //cin.get();
  return V;
}

Vector FtauForce(Vector const& X, Vector const& XG, Vector const& normal,
                 int dim, int tag, double theta, double Kforp, double nforp, double t,
                 Vector const& Vs, Matrix3d const& Q, int thetaDOF, double Kcte)//
{
  double x = X(0);
  double y = X(1);

  Vector tau(dim);
  tau(0) = +normal(1); tau(1) = -normal(0);  //this tangent goes in the direction of the parametrization
  //tau = -tau;;//this tangent goes against the parametrization
  int cas = 100;
  Vector f(Vector::Zero(X.size()));

  if (cas == 0 && dim == 2)//////////////////////////////////////////////////
  {//don't forget to put viscosity 1 for the unitary circular, spherical toy case
    double psi = 0.0;
    double B1 = 1.0, B2 = -5.0, mu = muu(0), R = 1.0;
    psi = atan2PI(X(1)-XG(1),X(0)-XG(0));
    double uthe = (-mu/R) * (2*B1*sin(theta-psi) + 5*B2*sin(theta-psi)*cos(theta-psi));
    //f(0) = k*normal(0); f(1) = k*normal(1);
    f = uthe*tau;
    //if ( X(1)-XG(1) < 0.0 )
    //  f = Vector::Zero(dim);
    //else
    //  f = f*(X(1)-XG(1))/0.2;
    //f(0) = 0; f(1) = -0.1;
  }
  else if (cas == 1)//////////////////////////////////////////////////
  {
    double psi = 0.0, k = 1.0;
    double uthe = 1.0/sqrt(k + Vs.dot(tau)*Vs.dot(tau));
    f = uthe*tau;
  }
  else if (cas == 2 && dim == 2)//////////////////////////////////////////////////
  {// For metachronal wave-like force
    Matrix3d Qr(Matrix3d::Zero(3,3));
    Vector3d Xref(Vector3d::Zero(3)), X3(Vector::Zero(3));
    X3(0) = X(0); X3(1) = X(1);
    Xref = RotM(theta,Qr,dim).transpose()*(X3 - XG); //cout << Xref.transpose() << "  " << X3.transpose() << "  " << XG.transpose() << endl;
    double uthe = 0.0;
    //double a = 2.0e-0, Tp = 0.05, omega = -2*pi/Tp, eta = 10;
    double a = 110.0e-0, Tp = 0.2, omega = -2*pi/Tp, eta = 10;
    double W = S_arcl(Xref(0), 0.0), S;  //cout << S_arcl(Xref(0), 0.0) << "  " << S_arcl(a, 0.0) << "  " << S_arcl(-a, 0.0) << endl;
    double L = S_arcl(-a,0.0);  //cout << "for " << Xref(0) << "  " << W << "  " << L << endl;
    double K = 1e-1*Kforp*L;  //0.015*L;
    double n = nforp /*(double)tag*/, k = 2*pi/L * n;

    S = W;
    for (int ni = 0; ni < 100; ni++){
      double F0 = S + K*tanh(eta*sin(pi*S/L))*cos(k*S-omega*t) - W;
      double F1 = 1 + (pi*K*eta/L)*(1 - pow(tanh(eta*sin(pi*S/L)),2.0))*cos(pi*S/L)*cos(k*S-omega*t)
                  - k*K*tanh(eta*sin(pi*S/L))*sin(k*S-omega*t);
      S = S - F0/F1;
    }

    if (Xref(1) <= 0){
      uthe = K*tanh(eta*sin(pi*S/L))*omega*sin(k*S - omega*t);
    }
    else{
      uthe = -K*tanh(eta*sin(pi*S/L))*omega*sin(k*S - omega*t);
    }
    f = uthe*tau;  //cout << V.transpose() << endl;
  }
  else if (cas == 3)//////////////////////////////////////////////////
  {//for tangent force proportional to slip_vel - vmetachronal (is vmet=0, then remains only slip_vel)
    double a = 110.0e-0;
    double L = S_arcl(-a,0.0);
    double k = Kcte*muu(0)/L;
    Vector Vmet = SlipVel(X, XG, normal, dim, tag, theta, Kforp, nforp, t, Q, thetaDOF);
    double uthe = k*(Vs.dot(tau)-Vmet.dot(tau));
    f = uthe*tau;  //cout << f.transpose() << endl;
  }
  else if (cas == 4)//////////////////////////////////////////////////
  {
    double a = 110.0e-0;
    double L = S_arcl(-a,0.0);
    double k = Kcte; //Kcte*muu(0)/L;
    Vector Vmet = SlipVel(X, XG, normal, dim, tag, theta, Kforp, nforp, t, Q, thetaDOF);
    double uthe = k*(Vmet.dot(tau));
    f = uthe*tau;
  }

  return f;
}

double DFtauForce(Vector const& X, Vector const& XG, Vector const& normal,
                  int dim, int tag, double theta, double Kforp, double nforp, double t,
                  Vector const& Vs, double Kcte)
{
  double a = 110.0e-0;
  double L = S_arcl(-a,0.0);
  double k = 0*Kcte; //Kcte*muu(0)/L;
  Vector tau(dim);
  tau(0) = +normal(1); tau(1) = -normal(0);  //tau(0) = -normal(1); tau(1) = +normal(0);
  double duthe = 0.0;
  if (false){
    duthe = -Vs.dot(tau)/pow(k + Vs.dot(tau)*Vs.dot(tau),1.5);
  }
  if (true){
    duthe = k;
  }
  return duthe;//pow(2,1.5);//
}

/*
double Vector FtauForceCoef(Vector const& X, Vector const& XG, Vector const& normal,
    int dim, int tag, double theta, double Kforp, double nforp, double t,
    Vector const& Vs, Matrix3d const& Q, int thetaDOF, double Kcte)//
{



  f = uthe*tau;  //cout << V.transpose() << endl;
}
*/

VectorXi DOFS_elimination(int LZ)
{ //0 for component to eliminate, 1 for component to compute
  VectorXi s_DOFS(LZ);
  if (LZ > 3)
    s_DOFS << 0, 1, 0, 1;
  else
    s_DOFS << 0, 1, 0;
  return s_DOFS;
}

double Dif_coeff(int tag)
{
  return 1;
}
double nuB_coeff(int tag)
{/*
  for (int i = 121; i < 141; i++){
    if (tag == i){
      return 1;
    }
  }
  return 0;*/
  /*
  if (tag == 102)
    return 1;
  else
    return 0;*/
  if (tag == 103 || tag == 104)
    return 1;
  else
    return 0;
}
double sig_coeff(int tag)
{
  return 10;
}
double bbG_coeff(Vector const& X, int tag)
{
  return 1;
}
double per_Elect(int tag)
{
  return 1;
}

Vector traction_maxwell(Vector const& E, Vector const& normal, double eps, int tag)
{
  int d = E.size();
  Vector T(Vector::Zero(3));
  Tensor I(Tensor::Identity(3,3));
  Vector Ec(Vector::Zero(3)), normalc(Vector::Zero(3));
  if (d == 2){
    Ec(0) = E(0); Ec(1) = E(1); normalc(0) = normal(0); normalc(1) = normal(1);
  }
  else{
    Ec = E; normalc = normal;
  }
  T = eps*(Ec*Ec.transpose() - Ec.norm()*Ec.norm()*I)*normal;
  return T;
}

Vector exact_tangent_ellipse(Vector const& X, Vector const& Xc, double theta, double R1, double R2, int dim)
{
  Vector Xcan(Vector::Zero(3));
  Vector Tan(Vector::Zero(dim));
  Vector X3(Vector::Zero(3));
  X3(0) = X(0); X3(1) = X(1);
  Xcan = RotM(-theta, dim)*(X3-Xc);
  Tan(0) = -R1*Xcan(1)/R2;
  Tan(1) =  R2*Xcan(0)/R1;
  Tan.normalize();
  return Tan;
}

Vector exact_normal_ellipse(Vector const& X, Vector const& Xc, double theta, double R1, double R2, int dim)
{
  Vector Xcan(Vector::Zero(3));
  Vector Nrm(Vector::Zero(dim));
  Vector X3(Vector::Zero(3));
  X3(0) = X(0); X3(1) = X(1);
  Xcan = RotM(-theta, dim)*(X3-Xc);
  Nrm(0) = R2*Xcan(0)/R1;
  Nrm(1) = R1*Xcan(1)/R2;
  Nrm.normalize();
  return Nrm;
}

Vector cubic_ellipse(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim)
{
  Vector Phib(Vector::Zero(dim));
  Phib = X0 + yb*(T0) + yb*yb*(3*X2-3*X0-2*T0-T2) + yb*yb*yb*(2*X0-2*X2+T0+T2);
  return Phib;
}

Vector Dcubic_ellipse(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim)
{
  Vector DPhib(Vector::Zero(dim));
  DPhib = T0 + 2*yb*(3*X2-3*X0-2*T0-T2) + 3*yb*yb*(2*X0-2*X2+T0+T2);
  return DPhib;
}

Vector curved_Phi(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim)
{
  Vector Phi(Vector::Zero(dim));
  if (false){
    if (yb != 1){
      Phi = (1.0/(1.0-yb))*(cubic_ellipse(yb,X0,X2,T0,T2,dim)+yb*(X0-X2)-X0);}
    else{
      Phi = -Dcubic_ellipse(1.0,X0,X2,T0,T2,dim)-X0+X2;}
  }
  else{//using T0 as Xc and T2 as a container for R1,R2 and theta
    Vector Xc = T0;
    double R1 = T2(0), R2 = T2(1), theta = T2(2);  //cout << Xc.transpose() << "   " << R1 << "   " << R2 << "   " << theta << endl;
    if (yb != 1){
      Phi = (1.0/(1.0-yb))*(exact_ellipse(yb,X0,X2,Xc,theta,R1,R2,dim)+yb*(X0-X2)-X0);}
    else{
      Phi = -Dexact_ellipse(1.0,X0,X2,Xc,theta,R1,R2,dim)-X0+X2;}
  }
  return Phi;
}

Vector Dcurved_Phi(double yb, Vector const& X0, Vector const& X2, Vector const& T0, Vector const& T2, int dim)
{
  Vector DPhi(Vector::Zero(dim));
  if (false){
    if (yb != 1){
      DPhi = (1.0/((1.0-yb)*(1.0-yb)))*((Dcubic_ellipse(yb,X0,X2,T0,T2,dim)+X0-X2)*(1-yb)
             +cubic_ellipse(yb,X0,X2,T0,T2,dim)+yb*(X0-X2)-X0);}
    else{
      DPhi = -Dcubic_ellipse(1.0,X0,X2,T0,T2,dim)-X0+X2;}
  }
  else{//using T0 as Xc and T2 as a container for R1,R2 and theta
    Vector Xc = T0;
    double R1 = T2(0), R2 = T2(1), theta = T2(2);
    if (yb != 1){
      DPhi = (1.0/((1.0-yb)*(1.0-yb)))*((1-yb)*(Dexact_ellipse(yb,X0,X2,Xc,theta,R1,R2,dim)+X0-X2)
             +exact_ellipse(yb,X0,X2,Xc,theta,R1,R2,dim)+yb*(X0-X2)-X0);}
    else{
      DPhi = -Dexact_ellipse(1.0,X0,X2,Xc,theta,R1,R2,dim)-X0+X2; cout << "what?" << endl;}
    Vector tmp(2); tmp << 8,8;
    //cout << (exact_ellipse(yb,X0,X2,Xc,theta,R1,R2,dim)-tmp).dot(Dexact_ellipse(yb,X0,X2,Xc,theta,R1,R2,dim)) << endl;
  }

  return DPhi;
}

double atan2PI(double a, double b) //calculates atan(a/b), a = y, b = x
{
  if (b == 0)
    if (a > 0)
      return pi/2.0;
    else
      return 3.0*pi/2.0;

  double s = atan(a/b);
  if (a >= 0 && b > 0)
    s = s + 0.0;
  else if (b < 0)
    s = s + pi;
  else if (a < 0 && b > 0)
    s = s + 2*pi;
  else
    throw;

  return s;
}

Vector exact_ellipse(double yb, Vector const& X0, Vector const& X2,
                     Vector const& Xc, double theta, double R1, double R2, int dim)
                     //Vector &Phib, Vector &DPhib)
{
  Vector Phib(Vector::Zero(dim));
  //Vector DPhib(Vector::Zero(dim));
  Vector Xcan0(Vector::Zero(3));
  Vector Xcan2(Vector::Zero(3));
  Vector X30(Vector::Zero(3));
  Vector X32(Vector::Zero(3));
  X30(0) = X0(0); X30(1) = X0(1);
  X32(0) = X2(0); X32(1) = X2(1);

  Xcan0 = RotM(-theta,dim)*X30-Xc;
  Xcan2 = RotM(-theta,dim)*X32-Xc;  //cout << endl << "Can Coords: " << Xcan0.transpose() << "   " << Xcan2.transpose();

  double s0 = atan2PI(R1*Xcan0(1),R2*Xcan0(0));
  double s2 = atan2PI(R1*Xcan2(1),R2*Xcan2(0));
  if (s0 < pi/2.0 && s2 > 3*pi/2.0){s0 = s0 + 2*pi;}
  else if (s2 < pi/2.0 && s0 > 3*pi/2.0){s2 = s2 + 2*pi;}
  double s  = (s2-s0)*yb+s0;  //cout << endl << "Param data:  " << s0 << "   " << s2 << "   " << s << endl;

  X30(0) = R1*cos(s)+Xc(0); X30(1) = R2*sin(s)+Xc(1);  //cout << X30.transpose() << endl;
  X30 = RotM(theta,dim)*X30;  //cout << X30.transpose() << endl;
  Phib(0) = X30(0); Phib(1) = X30(1);  //cout << Phib.transpose() << endl;

  //X30(0) = -R1*sin(s); X30(1) = R2*cos(s);
  //X30 = (s2-s0)*RotM(theta,dim)*X30;

  //DPhib(0) = X30(0); DPhib(1) = X30(1);

  return Phib;
}

Vector Dexact_ellipse(double yb, Vector const& X0, Vector const& X2,
                     Vector const& Xc, double theta, double R1, double R2, int dim)
                     //Vector &Phib, Vector &DPhib)
{
  //Vector Phib(Vector::Zero(dim));
  Vector DPhib(Vector::Zero(dim));
  Vector Xcan0(Vector::Zero(3));
  Vector Xcan2(Vector::Zero(3));
  Vector X30(Vector::Zero(3));
  Vector X32(Vector::Zero(3));
  X30(0) = X0(0); X30(1) = X0(1);
  X32(0) = X2(0); X32(1) = X2(1);

  Xcan0 = RotM(-theta,dim)*X30-Xc;
  Xcan2 = RotM(-theta,dim)*X32-Xc;

  double s0 = atan2PI(R1*Xcan0(1),R2*Xcan0(0));
  double s2 = atan2PI(R1*Xcan2(1),R2*Xcan2(0));
  if (s0 < pi/2.0 && s2 > 3*pi/2.0){s0 = s0 + 2*pi;}
  else if (s2 < pi/2.0 && s0 > 3*pi/2.0){s2 = s2 + 2*pi;}
  double s  = (s2-s0)*yb+s0;  //cout << endl << "Param data:  " << s0 << "   " << s2 << "   " << s << endl;

  //X30(0) = R1*cos(s)+Xc(0); X30(1) = R2*sin(s)+Xc(1);
  //X30 = RotM(theta,dim)*X30;

  //Phib(0) = X30(0); Phib(1) = X30(1);

  X30(0) = -R1*sin(s); X30(1) = R2*cos(s);
  X30 = (s2-s0)*RotM(theta,dim)*X30;

  DPhib(0) = X30(0); DPhib(1) = X30(1);  //cout << "b." << DPhib.transpose() << endl;

  return DPhib;
}

double Flink(double t, int Nl){
  double ome = pi/2.0;
  double pha = 0*90.0*pi/180.0;
  double alp = .3;
  double lmax = 1;
  double d = 0.0;
  int cas = 1;

  if (cas == 1){
    d = +lmax*(alp + ((1-alp)/2.0) * (cos(ome*t + (Nl)*pha) + 1.0));
    //if (Nl != 0 || Nl != 1){d = alp*lmax;}
  }
  else if (cas == 2){
    double P = 8;
    double tP = t - floor(t/P)*P;
    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*tP) + 1.0));}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = lmax*alp;}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*(tP-2.0)) + 1.0));}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = lmax;}
    }
    else{
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = lmax;}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*(tP-2.0)) + 1.0));}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = lmax*alp;}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*(tP-4.0)) + 1.0));}
    }
  }
  else if (cas == 3){
    double P = 8;
    double tP = t - floor(t/P)*P;
    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*tP) + 1.0));}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = lmax*alp;}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = lmax*alp;}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*(tP-4.0)) + 1.0));}
    }
    else{
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = lmax;}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*(tP-2.0)) + 1.0));}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = lmax*(alp + ((1-alp)/2.0) * (cos(ome*(tP-2.0)) + 1.0));}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = lmax;}
    }
  }
  else if (cas == 4){
    double P = 4;
    double tP = t - floor(t/P)*P, eps = (1-alp)*lmax, lmin = alp*lmax;
    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = lmax - eps/(P/4.0)*tP;}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = lmin;}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = lmin + eps/(P/4.0)*(tP-P/2.0);}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = lmax;}
    }
    else{
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = lmax;}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = lmax - eps/(P/4.0)*(tP-P/4.0);}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = lmin;}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = lmin + eps/(P/4.0)*(tP-3.0*P/4.0);}
    }
  }

  return d;
}

double DFlink(double t, int Nl){
  double ome = pi/2.0;
  double pha = 0*90.0*pi/180.0;
  double alp = .4;
  double lmax = 1;
  double d = 0.0;
  int cas = 7;

  if (cas == 1){
    d = -lmax*((1-alp)/2.0)*ome*sin(ome*t + (Nl)*pha);
    //if (Nl != 0 || Nl != 1){d = 0;}
    //if (Nl == 0 || Nl == 1){d = 0;}
  }
  else if (cas == 2){
    double P = 8;
    double tP = t - floor(t/P)*P;
    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*tP);}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = 0.0;}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-2.0));}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else{
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = 0.0;}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-2.0));}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = 0.0;}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-4.0));}
    }
  }
  else if (cas == 3){
    double P = 8;
    double tP = t - floor(t/P)*P;
    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*tP);}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = 0.0;}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = 0.0;}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-4.0));}
    }
    else{
      if     ((      0.0 <= tP)&&(tP < P/4.0    )){d = 0.0;}
      else if((    P/4.0 <= tP)&&(tP < P/2.0    )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-2.0));}
      else if((    P/2.0 <= tP)&&(tP < 3.0*P/4.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-2.0));}
      else if((3.0*P/4.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
  }
  else if (cas == 4){
    double P = 8;
    double tP = t - floor(t/P)*P, eps = (1-alp)*lmax, lmin = alp*lmax;
    if (Nl == 0){
      if     ((      0.0 <  tP)&&(tP <  P/4.0    )){d = - eps/(P/4.0);}
      else if((    P/4.0 <= tP)&&(tP <= P/2.0    )){d = 0.0;}
      else if((    P/2.0 <  tP)&&(tP <  3.0*P/4.0)){d = + eps/(P/4.0);}
      else if((3.0*P/4.0 <= tP)&&(tP <= P        )){d = 0.0;}
    }
    else{
      if     ((      0.0 <= tP)&&(tP <= P/4.0    )){d = 0.0;}
      else if((    P/4.0 <  tP)&&(tP <  P/2.0    )){d = - eps/(P/4.0);}
      else if((    P/2.0 <= tP)&&(tP <= 3.0*P/4.0)){d = 0.0;}
      else if((3.0*P/4.0 <  tP)&&(tP <  P        )){d = + eps/(P/4.0);}
    }
  }
  else if (cas == 5){
    double P = 8.0;
    double tP = t - floor(t/P)*P, eps = (1-alp)*lmax, lmin = alp*lmax;
    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = - eps/(P/8.0);}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = + eps/(P/8.0);}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 1){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = - eps/(P/8.0);}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = + eps/(P/8.0);}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 2){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = - eps/(P/8.0);}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = + eps/(P/8.0);}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 3){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = - eps/(P/8.0);}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = + eps/(P/8.0);}
    }
  }
  else if (cas == 6){
    double P = 8.0;
    double tP = t - floor(t/P)*P;
    ome = 8.0*pi/P;

    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-0.0*P/8.0));}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-3.0*P/8.0));}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 1){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-1.0*P/8.0));}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-4.0*P/8.0));}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 2){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-2.0*P/8.0));}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-5.0*P/8.0));}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 3){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-3.0*P/8.0));}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-6.0*P/8.0));}
    }
  }
  else if (cas == 7){
    double P = 8.0;
    double tP = t - floor(t/P)*P;
    ome = 8.0*pi/P;

    if (Nl == 0){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-0.0*P/8.0));}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-6.0*P/8.0));}
    }
    else if (Nl == 1){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-1.0*P/8.0));}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-5.0*P/8.0));}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 2){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-2.0*P/8.0));}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = 0.0;}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = 0.0;}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-4.0*P/8.0));}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
    else if (Nl == 3){
      if     ((      0.0 <= tP)&&(tP < 1.0*P/8.0)){d = 0.0;}
      else if((1.0*P/8.0 <= tP)&&(tP < 2.0*P/8.0)){d = 0.0;}
      else if((2.0*P/8.0 <= tP)&&(tP < 3.0*P/8.0)){d = 0.0;}
      else if((3.0*P/8.0 <= tP)&&(tP < 4.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-3.0*P/8.0));}
      else if((4.0*P/8.0 <= tP)&&(tP < 5.0*P/8.0)){d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP-3.0*P/8.0));}
      else if((5.0*P/8.0 <= tP)&&(tP < 6.0*P/8.0)){d = 0.0;}
      else if((6.0*P/8.0 <= tP)&&(tP < 7.0*P/8.0)){d = 0.0;}
      else if((7.0*P/8.0 <= tP)&&(tP < P        )){d = 0.0;}
    }
  }
  else if (cas == 8){
    double P = 8.0;
    double tP = t - floor(t/P)*P;
    ome = 8.0*pi/P;

    if (Nl == 0){
      d = -lmax*((1-alp)/2.0) * ome * sin(ome*(tP));
    }
  }


  return d;
}

Vector Fdrag(int LZ){
  double visc = 1.0, R = 1.0, v = 1.0/3.0;
  Vector F(Vector::Zero(LZ));
  //F(1) = 6.0*pi*visc*R*v;
  //F(1) = 6.0*pi;
  return F;
}

double Ellip_arcl_integrand(double zi){
  double R = 0;
  //if (true){
  //double a = 110.0e-0, ecc = 0.944, b = a*sqrt(1.0-ecc*ecc)/*36.2940e-0*/, eb = 0.090;
  double a = 110.0e-0, ecc = 0.500, b = a*sqrt(1.0-ecc*ecc)/*36.2940e-0*/, eb = 0.090;
    //double a = 2.0e-0, b = 1.0e-0, eb = 0.0;
    //cout << -zi/(a*a) << "  " << 1.0-zi*zi/(a*a) << " " << sqrt(1.0-zi*zi/(a*a)) << endl;
    R = sqrt(1.0*1.0 + b*b*pow(( (-zi/(a*a)) * 1.0/sqrt(1.0-zi*zi/(a*a)) - eb*(pi/a)*cos(pi*zi/a) ),2.0));
    //cout << a << " " << b << " " << R << endl;
  //}
  return R;
}

double S_arcl(double z, double zc){
  double a = 110.0e-0, S = 0.0;
  //double a = 2.0e-0, S = 0.0;
  double li = z - zc, ls = a;  //cout << li << "  " << ls << endl;
  if (ls-li < 1e-8){
    //cout << "return prem" << endl;
    return 0.0;
  }
  if (false){
    //Gauss quadrature (doesn't work very well, singularities)//////////////////////////////////////////////////
    int is = 16;
    double w16[16] = {0.1894506104550685,0.1894506104550685,0.1826034150449236,0.1826034150449236,
                      0.1691565193950025,0.1691565193950025,0.1495959888165767,0.1495959888165767,
                      0.1246289712555339,0.1246289712555339,0.0951585116824928,0.0951585116824928,
                      0.0622535239386479,0.0622535239386479,0.0271524594117541,0.0271524594117541};
    double x16[16] = {-0.0950125098376374,0.0950125098376374,-0.2816035507792589,0.2816035507792589,
                      -0.4580167776572274,0.4580167776572274,-0.6178762444026438,0.6178762444026438,
                      -0.7554044083550030,0.7554044083550030,-0.8656312023878318,0.8656312023878318,
                      -0.9445750230732326,0.9445750230732326,-0.9894009349916499,0.9894009349916499};
    //double w5[5] = {0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891};
    //double x5[5] = {0.0000000000000000,-0.5384693101056831,0.5384693101056831,-0.9061798459386640,0.9061798459386640};
    for (int i = 0; i < is; i++){
      double eval = (ls-li)/2.0 * x16[i] + (ls+li)/2.0;
      S = S + w16[i]*Ellip_arcl_integrand(eval);
    }
    S = abs((ls-li)/2.0 * S);
  }
  else{
    //Gauss Chebyshev (works very well, singularities)//////////////////////////////////////////////////
    int Nroots = 100;
    for (int i = 0; i < Nroots; i++){
      double Xc = cos(pi*(2.0*(i+1)-1)/(2.0*(double)Nroots));
      double eval = (ls-li)/2.0 * Xc + (ls+li)/2.0;  //cout << eval << "  " << Ellip_arcl_integrand(eval) << endl;
      S = S + Ellip_arcl_integrand(eval)*sqrt(1-Xc*Xc);  //cout << S << endl;
    }
    S = abs((ls-li)/2.0 * S) * pi/(double)Nroots;
    //cout << S << endl;
  }

  return S;
}

double ElastPotEner(double Kelast, double xi, double R){
  double Ep = 0.0;
  int cas = 1;
  if (cas == 0){//Elastic
    Ep = 0.5*Kelast*xi*xi;
  }
  else if(cas == 1){
    double h = R*pow((1+xi)/(1-xi),2.0/3.0), w = R*pow((1-xi)/(1+xi),1.0/3.0);
    if (xi < 0){//Oblate
      double e = sqrt(1-h*h/(w*w));
      Ep = 2*pi*w*w*(1 + (h*h/(e*w*w))* atanh(e));
    }
    else if (xi > 0){//Prolate
      double e = sqrt(1-w*w/(h*h));
      Ep = 2*pi*w*w*(1 + (h/(e*w))* asin(e));
    }
    else{
      Ep = 4*pi*R*R;
    }
    Ep = Kelast*Ep;
  }

  return Ep;
}

double DElastPotEner(double Kelast, double xi, double R){
  double DEp = 0.0;
  int cas = 1;
  if (cas == 0){//Elastic
    DEp = Kelast*xi;
  }
  else if(cas == 1){//Surface tension
    //double h = R*pow((1.0+xi)/(1.0-xi),2.0/3.0), w = R*pow((1.0-xi)/(1.0+xi),1.0/3.0);
    if (xi < 0){//Oblate
      double sig = atanh(2.0*sqrt(-xi)/(xi-1.0));
      DEp = -2.0*pi*R*R*pow(1.0+xi,-2.0/3.0)*pow(1.0-xi,2.0/3.0)*( (1.0+xi)/(2.0*xi*(xi-1))
              + sig*pow(1.0+xi,3.0)/(4.0*pow(-xi,3.0/2.0)*pow(xi-1.0,2.0)) + 2.0*sig*(1.0+xi)/(sqrt(-xi)*pow(xi-1.0,2.0)) )
            -(8.0/3.0)*pi*R*R*pow(1.0-xi,-1.0/3.0)*pow(1.0+xi,-5.0/3.0)*( 1.0 + sig*pow(1.0+xi,2.0)/(2.0*sqrt(-xi)*(xi-1.0)) );
    }
    else if (xi > 0){//Prolate
      double sig = asin(2.0*sqrt(xi)/(1.0+xi));
      DEp = -2.0*pi*R*R*pow(1.0+xi,-2.0/3.0)*pow(1.0-xi,2.0/3.0)*( (1.0+xi)/(2.0*xi*(xi-1.0))
              + sig*(1.0+xi)/(4.0*pow(xi,3.0/2.0)) - sig*(1.0+xi)/(sqrt(xi)*pow(xi-1.0,2.0)) )
            -(8.0/3.0)*pi*R*R*pow(1.0-xi,-1.0/3.0)*pow(1.0+xi,-5.0/3.0)*( 1.0 - sig*pow(1.0+xi,2.0)/(2.0*sqrt(xi)*(xi-1.0)) );
    }
    else{//Sphere
      DEp = 0.0;
    }
    DEp = Kelast*DEp;
  }

  return DEp;
}

double DDElastPotEner(double Kelast, double xi, double R){
  double DDEp = 0.0;
  int cas = 1;
  if (cas == 0){//Elastic
    DDEp = Kelast;
  }
  else if(cas == 1){
    //double h = R*pow((1+xi)/(1-xi),2.0/3.0), w = R*pow((1-xi)/(1+xi),1.0/3.0);
    if (xi < 0){//Oblate
      double sig1 = atanh(2*sqrt(-xi)/(xi-1.0)), sig2 = pow(-xi,11.0/2.0);
      DDEp = -pi*R*R/(36.0*sig2*pow(1.0-xi,13.0/3.0)*pow(1.0+xi,11.0/3.0)) * ( 54.0*pow(-xi,7.0/2.0)+84.0*pow(-xi,9.0/2.0)
              -308.0*pow(-xi,7.0/2.0)*pow(-xi,11.0/2.0)-700.0*pow(-xi,13.0/2.0)-168.0*pow(-xi,15.0/2.0)+508.0*pow(-xi,17.0/2.0)
              +404.0*pow(-xi,19.0/2.0)+108.0*pow(-xi,21.0/2.0)+18.0*pow(-xi,23.0/2.0)
              -27.0*pow(xi,3.0)*sig1+33.0*pow(xi,4.0)*sig1+68.0*pow(xi,5.0)*sig1-172.0*pow(xi,6.0)*sig1-106.0*pow(xi,7.0)*sig1
              +254.0*pow(xi,8.0)*sig1+116.0*pow(xi,9.0)*sig1-124.0*pow(xi,10.0)*sig1-51.0*pow(xi,11.0)*sig1+9.0*pow(xi,12.0)*sig1 );
    }
    else if (xi > 0){//Prolate
      double sig = asin(2.0*sqrt(xi)/(1.0+xi));
      DDEp = +pi*R*R/(36.0*pow(xi,5.0/2.0)*pow(1.0-xi,7.0/3.0)*pow(1.0+xi,8.0/3.0)) * ( 27*sig-47.0*pow(xi,2.0)*sig+92.0*pow(xi,3.0)*sig
              +157.0*pow(xi,4.0)*sig+42.0*pow(xi,5.0)*sig-9.0*pow(xi,6.0)*sig-6.0*xi*sig
              -54.0*pow(xi,1.0/2.0)+30.0*pow(xi,3.0/2.0)+284.0*pow(xi,5.0/2.0)-332.0*pow(xi,7.0/2.0)+90.0*pow(xi,9.0/2.0)-18.0*pow(xi,11.0/2.0) );
    }
    else{//Sphere
      DDEp = 256.0*pi*R*R/45.0;
    }
    DDEp = Kelast*DDEp;
  }

  return DDEp;
}

#endif
