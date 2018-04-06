#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_DONT_PARALLELIZE
#include <Fepic/Mesh>
#include <Fepic/Quadrature>
#include <Fepic/DofHandler>
#include <Fepic/Shape>
#include <Eigen/LU>
#include <Eigen/Dense>
#include "petscsnes.h"
#include <iostream>
#include <tr1/memory>
#include <tr1/array>
#include "mypetsc.hpp"
#include <sys/types.h>  //for mkdir
#include <sys/stat.h>   //for check mkdir
#include <unistd.h>     //for mkdir

using namespace std;
using namespace Eigen;
using namespace tr1;

// por célula
#define MAX_DOFS_U 33 // P2hp
#define MAX_DOFS_P 4  // P1
#define MAX_NNODES 10 // P2

enum VarNumber {
  // DH_UNKS
  VAR_U = 0,
  VAR_P = 1,
  VAR_Z = 2,  //variable solid

  // DH_MESH
  VAR_M = 0,

  // DH_SQRM
  VAR_S = 0
};

enum DofHandlerNumber {
  DH_UNKM = 0,
  DH_MESH = 1,
  DH_SLIP = 2
};

enum PairSpace {
  P1P1      = 1,
  P1bP1_c   = 2,
  P2bPm1_c  = 3,
  P2P1      = 4,
  P1bP1     = 5,
  P2P0      = 6,
  P2bPm1    = 7,
  P1P1unstable = 8,
  P2bP1    = 9
};

const double pi  = 3.141592653589793;
const double pi2 = pi*pi;

typedef Matrix<VectorXd, Dynamic,1>  VecOfVec;
typedef Matrix<MatrixXd, Dynamic,1>  VecOfMat;

// space
typedef Matrix<double, Dynamic,1,0,6,1>              Vector;  //6 is _MaxRows in the dynamic option
typedef Matrix<double, Dynamic,Dynamic,RowMajor,3,3> Tensor;
typedef Matrix<double, 3, 3>                         Tensor3;
typedef Matrix<double, Dynamic,Dynamic,RowMajor,6,6> TensorZ;
typedef Matrix<int, Dynamic,Dynamic,RowMajor,6,6>    TensorXi;

template<class Vec, class T>
bool is_in(T value, Vec const& v)
{
  for (int i = 0; i < (int)v.size(); i++)
  {
    if (value==v[i])
      return true;
  }
  return false;
}

template<class Vec, class T>
int is_in_id(T value, Vec const& v)
{
  for (int i = 0; i < (int)v.size(); i++)
  {
    if (value==v[i])
      return i+1;
  }
  return 0;
}

/* stabilization type */
enum Behaviors {
  BH_bble_condens_PnPn = 0x01,
  BH_GLS               = 0x02,
  BH_Press_grad_elim   = 0x04,
  BH_bble_condens_CR   = 0x08
};

class AppCtx;
class Statistics;

double pho(Vector const& X, int tag);
double gama(Vector const& X, double t, int tag);
double cos_theta0();
double zeta(double u_norm, double angle);
double beta_diss();
double muu(int tag);
Vector force(Vector const& X, double t, int tag);
Vector u_exact(Vector const& X, double t, int tag);
Vector z_exact(Vector const& X, double t, int tag);
Vector traction(Vector const& X, Vector const& normal, double t, int tag);
double p_exact(Vector const& X, double t, int tag);
Vector grad_p_exact(Vector const& X, double t, int tag);
Tensor grad_u_exact(Vector const& X, double t, int tag);
Vector u_initial(Vector const& X, int tag);
double p_initial(Vector const& X, int tag);
Vector z_initial(Vector const& X, int tag);
Vector solid_normal(Vector const& X, double t, int tag);
Vector v_exact(Vector const& X, double t, int tag);
Vector solid_veloc(Vector const& X, double t, int tag);
Tensor feature_proj(Vector const& X, double t, int tag);
Vector gravity(Vector const& X, int dim);
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
TensorZ MI_tensor(double M, double R, int dim, Tensor3 TI);
Matrix3d RotM(double theta, int dim);
//Vector SlipVel(Vector const& X, Vector const& XG, int dim, int tag);
Vector SlipVel(Vector const& X, Vector const& XG, Vector const& normal, int dim, int tag, double theta);
Vector force_Htau(Vector const& X, Vector const& XG, Vector const& normal, int dim, int tag, double theta);

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

inline double sqr(double v) {return v*v;}

void inline inverseAndDet(Tensor const& a, int dim, Tensor& inv, double& det)
{
  if (dim==2)
  {
    det = a(0,0)*a(1,1)-a(0,1)*a(1,0);

    inv(0,0) = a(1,1)/det;
    inv(0,1) = -a(0,1)/det;
    inv(1,0) = -a(1,0)/det;
    inv(1,1) = a(0,0)/det;
  }
  else if (dim==3)
  {
    det = a(0,0)*(a(1,1)*a(2,2)-a(1,2)*a(2,1))+a(0,1)*(a(1,2)*a(2,0)-a(1,0)*a(2,2))+a(0,2)*(a(1,0)*a(2,1)-a(1,1)*a(2,0));

    inv(0,0) = ( a(1,1)*a(2,2)-a(1,2)*a(2,1) )/det;
    inv(0,1) = ( a(0,2)*a(2,1)-a(0,1)*a(2,2) )/det;
    inv(0,2) = ( a(0,1)*a(1,2)-a(0,2)*a(1,1) )/det;
    inv(1,0) = ( a(1,2)*a(2,0)-a(1,0)*a(2,2) )/det;
    inv(1,1) = ( a(0,0)*a(2,2)-a(0,2)*a(2,0) )/det;
    inv(1,2) = ( a(0,2)*a(1,0)-a(0,0)*a(1,2) )/det;
    inv(2,0) = ( a(1,0)*a(2,1)-a(1,1)*a(2,0) )/det;
    inv(2,1) = ( a(0,1)*a(2,0)-a(0,0)*a(2,1) )/det;
    inv(2,2) = ( a(0,0)*a(1,1)-a(0,1)*a(1,0) )/det;

  }
  else
  {
    printf("invalid dim\n");
    throw;
  }
}

double inline determinant(Tensor const& a, int dim)
{
  if (dim==1)
    return a(0,0);
  else
  if (dim==2)
    return a(0,0)*a(1,1)-a(0,1)*a(1,0);
  else
  if (dim==3)
    return a(0,0)*(a(1,1)*a(2,2)-a(1,2)*a(2,1))+a(0,1)*(a(1,2)*a(2,0)-a(1,0)*a(2,2))+a(0,2)*(a(1,0)*a(2,1)-a(1,1)*a(2,0));
  else
  {
    printf("double determinant(Tensor const& a, int dim): invalid dim, get %d\n", dim);
    throw;
  }
}

template<class TensorType>
void invert(TensorType & a, int dim)
{
  if (dim==1)
  {
    a(0,0)=1./a(0,0);
  }
  else
  if (dim==2)
  {
    double const det = a(0,0)*a(1,1)-a(0,1)*a(1,0);

    double const inv00 = a(1,1)/det;
    double const inv01 = -a(0,1)/det;
    double const inv10 = -a(1,0)/det;
    double const inv11 = a(0,0)/det;

    a(0,0) = inv00;
    a(0,1) = inv01;
    a(1,0) = inv10;
    a(1,1) = inv11;
  }
  else if (dim==3)
  {
    double const det = a(0,0)*(a(1,1)*a(2,2)-a(1,2)*a(2,1))+a(0,1)*(a(1,2)*a(2,0)-a(1,0)*a(2,2))+a(0,2)*(a(1,0)*a(2,1)-a(1,1)*a(2,0));

    double const inv00 = ( a(1,1)*a(2,2)-a(1,2)*a(2,1) )/det;
    double const inv01 = ( a(0,2)*a(2,1)-a(0,1)*a(2,2) )/det;
    double const inv02 = ( a(0,1)*a(1,2)-a(0,2)*a(1,1) )/det;
    double const inv10 = ( a(1,2)*a(2,0)-a(1,0)*a(2,2) )/det;
    double const inv11 = ( a(0,0)*a(2,2)-a(0,2)*a(2,0) )/det;
    double const inv12 = ( a(0,2)*a(1,0)-a(0,0)*a(1,2) )/det;
    double const inv20 = ( a(1,0)*a(2,1)-a(1,1)*a(2,0) )/det;
    double const inv21 = ( a(0,1)*a(2,0)-a(0,0)*a(2,1) )/det;
    double const inv22 = ( a(0,0)*a(1,1)-a(0,1)*a(1,0) )/det;

    a(0,0) = inv00;
    a(0,1) = inv01;
    a(0,2) = inv02;
    a(1,0) = inv10;
    a(1,1) = inv11;
    a(1,2) = inv12;
    a(2,0) = inv20;
    a(2,1) = inv21;
    a(2,2) = inv22;

  }
  else
  {
    printf("invalid dim, try to run again dumb \n");
    throw;
  }


}

inline Vector cross(Vector const& a, Vector const& b)
{
  Vector c(a.size());
  
  c(0) = a(1)*b(2) - a(2)*b(1);
  c(1) = a(2)*b(0) - a(0)*b(2);
  c(2) = a(0)*b(1) - a(1)*b(0);

  return c;
}
inline void cross(Vector & c, Vector const& a, Vector const& b)
{
  c(0) = a(1)*b(2) - a(2)*b(1);
  c(1) = a(2)*b(0) - a(0)*b(2);
  c(2) = a(0)*b(1) - a(1)*b(0);

}

inline Vector SolidVel(Vector const& X, Vector const& Xg, Vector const& Z, int dim){
  Vector R(dim);
  if (dim == 2){
    R(0) = Z(0) - Z(2)*(X(1)-Xg(1));
    R(1) = Z(1) + Z(2)*(X(0)-Xg(0));
  }
  else if(dim == 3){
    R = Z.head(3) + cross(Z.tail(3),X-Xg);
  }
  return R;
}

inline TensorZ SolidVelMatrix(Vector const& X, Vector const& Xg, int dim, int LZ){
  TensorZ H = TensorZ::Zero(dim,LZ);
  if (dim == 2){
    H(0,0) = 1.0; H(1,1) = 1.0;
    H(0,2) = -(X(1)-Xg(1));
    H(1,2) =  (X(0)-Xg(0));
  }
  else if(dim == 3){
    //H = Z.head(3) + cross(Z.tail(3),X-Xg);
  }
  return H;
}

inline bool lessVector(Vector2d const& a, Vector2d const& b){
  return (a(0) < b(0)) || ((a(0) == b(0)) && (a(1) < b(1)));
}
inline double cross2d(Vector2d const& o, Vector2d const& a, Vector2d const& b){
  return (a(0)-o(0))*(b(1)-o(1)) - (a(1)-o(1))*(b(0)-o(0));
}

inline double DobCont(Tensor const& A, Tensor const& B){
  double d = 0;
  for (int i = 0; i < A.rows(); i++) {d += A.row(i).dot(B.row(i));}
  return d;
}

inline std::vector<Vector3d> midGP(std::vector<Vector3d> XG, std::vector<Vector3d> XG_0, double utheta, int N){
  std::vector<Vector3d> XG_mid(N);
  for (int i = 0; i < N; i++){XG_mid[i] = utheta*XG[i] + (1.-utheta)*XG_0[i];}
  return XG_mid;
}


class Statistics
{
  template<class T>
  T max(T const& a, T const& b)
  {
    return a<b?b:a;
  }
  
public:

  Statistics() : p_L2_norm        (_data[0]),
                 u_L2_norm        (_data[1]),
                 grad_u_L2_norm   (_data[2]),
                 grad_p_L2_norm   (_data[3]),
                 p_inf_norm       (_data[4]),
                 u_inf_norm       (_data[5]),
                 hmean            (_data[6]),
                 u_L2_facet_norm  (_data[7]),
                 u_inf_facet_norm (_data[8])
  {
    this->reset();
  }
  
  void reset()
  {
    for (int i = 0; i < _n_data; ++i)
      _data[i] = 0;
  }

  /* stores the highest value */

#define STORES_HIGHEST(name) void add_##name(double x)      \
                             {                              \
                               name = max(x, name);         \
                             }

  STORES_HIGHEST(p_L2_norm)
  STORES_HIGHEST(u_L2_norm)
  STORES_HIGHEST(grad_u_L2_norm)
  STORES_HIGHEST(grad_p_L2_norm)
  STORES_HIGHEST(hmean)
  STORES_HIGHEST(p_inf_norm)
  STORES_HIGHEST(u_inf_norm)
  STORES_HIGHEST(u_L2_facet_norm)
  STORES_HIGHEST(u_inf_facet_norm)

#undef STORES_HIGHEST

  double _data[9];
  static const int _n_data = sizeof(_data)/sizeof(double);

  double & p_L2_norm;
  double & u_L2_norm;
  double & grad_u_L2_norm;
  double & grad_p_L2_norm;
  double & p_inf_norm;
  double & u_inf_norm;
  double & hmean;
  double & u_L2_facet_norm;
  double & u_inf_facet_norm;


};



//
// User Class
//
class AppCtx
{
public:
  AppCtx(int argc, char **argv, bool &help_return, bool &erro);

  bool err_checks();
  void loadMesh();

  void loadDofs();
  void setUpDefaultOptions();
  bool getCommandLineOptions(int argc, char **/*argv*/);
  bool createFunctionsSpace();
  void createQuadrature();
  void meshAliasesUpdate();
  void dofsCreate();
  void dofsUpdate();
  PetscErrorCode allocPetscObjs();
  void matrixColoring();
  void printMatlabLoader();
  // must be called after loadDofs
  void evaluateQuadraturePts();
  void onUpdateMesh();
  PetscErrorCode setInitialConditions();
  PetscErrorCode checkSnesConvergence(SNES snes, PetscInt it,PetscReal xnorm, PetscReal pnorm, PetscReal fnorm, SNESConvergedReason *reason);
  PetscErrorCode setUPInitialGuess();
  PetscErrorCode solveTimeProblem();
//  PetscErrorCode formJacobian(SNES /*snes*/,Vec x,Mat *Mat_Jac, Mat* /*prejac*/, MatStructure * /*flag*/);
//  PetscErrorCode formFunction(SNES /*snes*/, Vec x, Vec f);

  PetscErrorCode formJacobian_fs(SNES /*snes*/,Vec x,Mat *Mat_Jac, Mat* /*prejac*/, MatStructure * /*flag*/);
  PetscErrorCode formFunction_fs(SNES /*snes*/, Vec x, Vec f);
  
  PetscErrorCode meshAdapt();
  
  PetscErrorCode formJacobian_mesh(SNES /*snes*/,Vec x, Mat *Mat_Jac, Mat* /*prejac*/, MatStructure * /*flag*/);
  PetscErrorCode formFunction_mesh(SNES /*snes*/, Vec x, Vec f);

  PetscErrorCode formJacobian_sqrm(SNES /*snes*/,Vec x, Mat *Mat_Jac, Mat* /*prejac*/, MatStructure * /*flag*/);
  PetscErrorCode formFunction_sqrm(SNES /*snes*/, Vec x, Vec f);
  
  // form the residue of the cell
  void formCellFunction(cell_iterator &cell,
                                  VectorXi &mapU_c,  VectorXi &/*mapP_c*/, // mappers
                                  MatrixXd &u_coefs_c_new,  VectorXd &p_coefs_c, // coefficients
                                  VectorXd &FUloc, VectorXd &FPloc); // output: local residue
  // form the residue of the facet
  void formFacetFunction(facet_iterator &facet,
                         VectorXi const&/*mapU_f*/,  VectorXi const&/*mapP_f*/, // mappers
                         MatrixXd &u_coefs_f,  VectorXd &/*p_coefs_f*/, // coefficients
                         VectorXd &FUloc); // output: local residue
  // form the residue of the contact line
  void formCornerFunction(CellElement *corner,
                          VectorXi const&/*mapU_r*/,  VectorXi const&/*mapP_r*/, // mappers
                          MatrixXd &u_coefs_r, // coefficients
                          VectorXd &FUloc);


  // get points dofs even if he lie at an edge
  /// @param[out] dofs
  void getNodeDofs(Point const* point, DofHandlerNumber DH_, VarNumber VAR_, int * dofs) const
  {
    if (mesh->isVertex(point))
    {
      dof_handler[DH_].getVariable(VAR_).getVertexAssociatedDofs(dofs, point);
    }
    else
    {
      const int m = point->getPosition() - mesh->numVerticesPerCell();
      Cell const* cell = mesh->getCellPtr(point->getIncidCell());
      if (dim==3)
      {
        const int edge_id = cell->getCornerId(m);
        dof_handler[DH_].getVariable(VAR_).getCornerAssociatedDofs(dofs, mesh->getCornerPtr(edge_id));
      }
      else
      {
        const int edge_id = cell->getFacetId(m);
        dof_handler[DH_].getVariable(VAR_).getFacetAssociatedDofs(dofs, mesh->getFacetPtr(edge_id));
      }
    }
  }

  void pressureTimeCorrection(Vec &Vec_up_0, Vec &Vec_up_1, double a, double b);

  std::vector<Vector2d> ConvexHull2d(std::vector<Vector2d> & LI);  //counterclockwise
  Vector getAreaMassCenterSolid(int solid, double &A);
  void calcHmean(double &hmean, double &hmin, double &hmax);
  double getMeshVolume();
  double getMaxVelocity();
  bool proxTest(MatrixXd &Contact, MatrixXd &ContW, double const INF);
  void forceDirichlet();
  PetscErrorCode updateSolidVel();
  void getSolidVolume();
  void getSolidCentroid();
  void getSolidInertiaTensor();
  PetscErrorCode moveCenterMass(double vtheta);
  PetscErrorCode updateSolidMesh();
  PetscErrorCode velNoSlip(Vec const& Vec_uzp, Vec const& Vec_sv, Vec &Vec_uzp_ns);
  PetscErrorCode plotFiles();
  Vector vectorSolidMesh(int const K, Point const* point, int const vs);
  void getFromBSV();
  PetscErrorCode saveDOFSinfo();
  Vector u_exacta(Vector const& X, double t, int tag);
  void computeForces(Vec const& Vec_x, Vec &Vec_up);
  //void printContactAngle(bool _print);

  void computeError(Vec const& Vec_x, Vec &Vec_up_1, double tt);
  void getVecNormals(Vec const* Vec_x_1, Vec & Vec_normal_);
  void smoothsMesh(Vec & Vec_normal_, Vec &Vec_x);
  void copyMesh2Vec(Vec &Vec_xmsh);
  void copyVec2Mesh(Vec const& Vec_xmsh);
  void swapMeshWithVec(Vec & Vec_xmsh);
  // @param[in] Vec_up_1 unknowns vector with fluid velocity
  // @param[out] u_mesh
  PetscErrorCode calcMeshVelocity(Vec const& Vec_x_0, Vec const& Vec_up_0, Vec const& Vec_up_1, double vtheta, Vec &Vec_v_mid, double tt); // by elasticity
  PetscErrorCode moveMesh(Vec const& Vec_x_0, Vec const& Vec_up_0, Vec const& Vec_up_1, double const vtheta, double tt, Vec & Vec_x_new);
  double getCellQuality(Vec const& Vec_x_, int cell_id) const;
  double getCellPatchQuality(Vec const& Vec_x_, int const* cells) const;
  void freePetscObjs();

  bool isFixedPoint(int tag) const
  {
    if (is_in(tag, neumann_tags) || is_in(tag, dirichlet_tags) || is_in(tag, periodic_tags))
      return true;
    else
      return false;
  }

  // global settings
  int         dim; // = space dim
  int         mesh_cell_type;   // ECellType
  int         function_space;  // P1P1, P2P1, P2P0, etc ...
  int         behaviors;
  int         n_modes;
  //double      Re;
  PetscBool   has_convec;
  PetscBool   unsteady;
  PetscBool   renumber_dofs;
  PetscBool   boundary_smoothing;
  PetscBool   print_to_matlab;
  PetscBool   force_dirichlet;
  PetscBool   full_diriclet;
  PetscBool   force_mesh_velocity;
  PetscBool   ale;
  PetscBool   plot_exact_sol;
  PetscBool   family_files;
  PetscBool   fprint_ca, fprint_hgv; // print contact angle, gravity velocity solid
  PetscBool   nonlinear_elasticity;
  PetscBool   mesh_adapt;
  PetscBool   time_adapt;
  PetscBool   is_mr_ab;
  PetscBool   is_bdf3;
  PetscBool   is_bdf2;
  PetscBool   is_bdf2_bdfe;
  PetscBool   is_bdf2_ab;
  PetscBool   is_bdf_cte_vel;
  PetscBool   is_bdf_euler_start;
  PetscBool   is_bdf_extrap_cte;
  PetscBool   is_basic;
  PetscBool   is_slipv;
  PetscBool   is_sslv;
  PetscBool   is_sfip;
  PetscBool   is_sfim;
  PetscBool   is_curvt;
  PetscBool   is_mr;
  
  int         converged_times;
  double      dt;
  double      steady_tol;
  double      utheta;
  double      vtheta;
  int         maxts;
  double      finaltime;
  PetscBool   force_pressure;
  bool        solve_the_sys;
  int         quadr_degree_cell;
  int         quadr_degree_facet;
  int         quadr_degree_corner;
  int         quadr_degree_err;
  bool        pres_pres_block;
  float       grow_factor;
  string      filename, filemass, filexas;
  string      filename_out, filehist_out;

  std::vector<int> dirichlet_tags;   // vetor com os tags que indicam contorno dirichlet
  std::vector<int> neumann_tags;     // vetor com os tags que indicam contorno neumann
  std::vector<int> interface_tags;
  std::vector<int> solid_tags;
  std::vector<int> triple_tags;
  std::vector<int> periodic_tags;
  std::vector<int> feature_tags; // 3d only .. assume 90 degree corners
  std::vector<int> flusoli_tags;  //fluid solid interface
  std::vector<int> fluidonly_tags;  //fluid only
  std::vector<int> solidonly_tags;  //solid only
  std::vector<int> slipvel_tags;

  DofHandler                   dof_handler[3];  //or 3
  MeshIoMsh                    msh_reader;
  MeshIoVtk                    vtk_printer;
  shared_ptr<Mesh>             mesh;
  // velocity
  shared_ptr<ShapeFunction>    shape_phi_c; // cell
  shared_ptr<ShapeFunction>    shape_phi_f; // facet
  shared_ptr<ShapeFunction>    shape_phi_r; // corner
  // pressure
  shared_ptr<ShapeFunction>    shape_psi_c; // cell
  shared_ptr<ShapeFunction>    shape_psi_f; // facet
  shared_ptr<ShapeFunction>    shape_psi_r; // corner
  // mesh
  shared_ptr<ShapeFunction>    shape_qsi_c; // cell
  shared_ptr<ShapeFunction>    shape_qsi_f; // facet
  shared_ptr<ShapeFunction>    shape_qsi_r; // corner

  shared_ptr<ShapeFunction>    shape_bble;



  shared_ptr<Quadrature>       quadr_cell;
  shared_ptr<Quadrature>       quadr_facet;
  shared_ptr<Quadrature>       quadr_corner;
  shared_ptr<Quadrature>       quadr_err;   // to compute error
  int                          n_qpts_cell;
  int                          n_qpts_facet;
  int                          n_qpts_corner;
  int                          n_qpts_err;

  // velocity
  VecOfVec                     phi_c;         // shape function evaluated at quadrature points
  VecOfVec                     phi_f;         // shape function evaluated at quadrature points (facet)
  VecOfVec                     phi_r;         // shape function evaluated at quadrature points (corner)
  // pressure
  VecOfVec                     psi_c;         // shape function evaluated at quadrature points
  VecOfVec                     psi_f;         // shape function evaluated at quadrature points (facet)
  VecOfVec                     psi_r;         // shape function evaluated at quadrature points (corner)
  // mesh
  VecOfVec                     qsi_c;         // shape function evaluated at quadrature points
  VecOfVec                     qsi_f;         // shape function evaluated at quadrature points (facet)
  VecOfVec                     qsi_r;         // shape function evaluated at quadrature points (corner)
  VectorXd                     qsi_c_at_center; // shape function evaluated at quadrature points


  // velocity
  VecOfMat                     dLphi_c;       // matriz de gradiente no elemento unitário
  VecOfMat                     dLphi_f;       // matriz de gradiente no elemento unitário (facet)
  VecOfMat                     dLphi_r;       // matriz de gradiente no elemento unitário (corner)
  // pressure
  VecOfMat                     dLpsi_c;       // matriz de gradiente no elemento unitário
  VecOfMat                     dLpsi_f;       // matriz de gradiente no elemento unitário (facet)
  VecOfMat                     dLpsi_r;       // matriz de gradiente no elemento unitário (corner)
  // mesh
  VecOfMat                     dLqsi_c;       // matriz de gradiente no elemento unitário
  VecOfMat                     dLqsi_f;       // matriz de gradiente no elemento unitário (facet)
  VecOfMat                     dLqsi_r;       // matriz de gradiente no elemento unitário (corner)
  // normal
  VecOfMat                     dLphi_nf;       // matriz de gradiente nos nós de uma facet
  // velocity, cell
  VectorXd                     bble;          // bubble function evaluated at quadrature points
  VecOfVec                     dLbble;

  VecOfMat                     dLqsi2_f;       // mesmo que dLqsi_f, aumentado de 1 grau (facet)
  
  //
  //          to compute error in each cell
  // velocity
  VecOfVec                     phi_err;         // shape function evaluated at quadrature points
  VecOfMat                     dLphi_err;       // matriz de gradiente no elemento unitário
  // pressure
  VecOfVec                     psi_err;         // shape function evaluated at quadrature points
  VecOfMat                     dLpsi_err;       // matriz de gradiente no elemento unitário
  // mesh
  VecOfVec                     qsi_err;         // shape function evaluated at quadrature points
  VecOfMat                     dLqsi_err;       // matriz de gradiente no elemento unitário


  // dofs
  int           n_unknowns;
  int           n_dofs_v_mesh;
  int           n_dofs_u_per_cell;
  int           n_dofs_u_per_facet;
  int           n_dofs_u_per_corner;
  int           n_dofs_p_per_cell;
  int           n_dofs_p_per_facet;
  int           n_dofs_p_per_corner;
  int           n_dofs_v_per_cell;
  int           n_dofs_v_per_facet;
  int           n_dofs_v_per_corner;
  int           n_dofs_s_per_cell;
  int           n_dofs_s_per_facet;
  int           n_dofs_s_per_corner;
  //int           n_dofs_v2_per_facet;
  int           n_unknowns_fs;
  int           n_unknowns_u;
  int           n_unknowns_p;
  int           n_unknowns_z;
  int           n_unknowns_sv;
  int           n_dofs_z_per_cell;
  int           n_dofs_z_per_facet;
  int           n_dofs_z_per_corner;
  int           n_unknowns_f;
  int           n_unknowns_s;
  int           n_nodes_fsi;
  int           n_nodes_fo;
  int           n_nodes_so;
  int           n_nodes_sv;
  
  int                    N_Solids, LZ;
  std::vector<int>       NN_Solids;
  std::vector<double>    MV, RV, VV;  //mass vector, radius vector, area vector
  std::vector<Vector3d>  XG_0, XG_1, XG_aux, XG_ini;
  double                 hme, hmn, hmx;
  std::vector<double>    theta_0, theta_1, theta_aux, theta_ini;
  std::vector<Tensor3>   InTen;
  std::vector<double>    RR, UU;
  //bool                   casevar = true, casevarc = true; //case variable or const to H solid vel functional

  // mesh alias
  int           n_nodes;
  int           n_cells;
  int           n_facets;
  int           n_corners;
  int           n_nodes_total;
  int           n_cells_total;
  int           n_facets_total;
  int           n_corners_total;
  int           nodes_per_cell;
  int           nodes_per_facet;
  int           nodes_per_corner;

  // solving ...
  double        current_time;
  int           time_step;
  int           print_step;
  double        beta1, beta2;

  Timer         timer;
  Statistics    Stats;

  // mesh size control ... mesh_size[i] is the mean of edge's size connected to node i at TIME=0
  std::vector<Real>    mesh_sizes;

  // fluid
  Vec                 Vec_res_fs, Vec_uzp_0, Vec_uzp_1;//, Vec_res, Vec_up_0, Vec_up_1,Vec_dup, /**/;
  Vec                 Vec_uzp_m1, Vec_uzp_m2; // for bdf3  //Vec_duzp_0, Vec_duzp,
  Mat                 Mat_Jac_fs;//, Mat_Jac;
  SNES                snes_fs;//, snes;         /* nonlinear solver context */
  KSP    			  ksp_fs;//ksp,
  PC	   			  pc_fs;//pc,
  SNESLineSearch      linesearch;

  // mesh
  Vec                 Vec_res_m,  Vec_v_mid, Vec_v_1, Vec_x_0, Vec_x_1, Vec_normal, Vec_tangent;
  Vec                 Vec_x_aux, Vec_x_cur; // bdf3
  Mat                 Mat_Jac_m;
  SNES                snes_m;
  KSP    			  ksp_m;
  PC	   			  pc_m;
  
  // slip velocity
  Vec                 Vec_slipv_0, Vec_slipv_1, Vec_slipv_m1, Vec_slipv_m2, Vec_normal_aux;

  // swimmer
  Vec                 Vec_res_s, Vec_slip_rho, Vec_rho_aux;  //Vec_uzp_0_ns, Vec_uzp_1_ns,;
  Mat                 Mat_Jac_s;
  SNES                snes_s;
  KSP                 ksp_s;
  PC                  pc_s;
  SNESLineSearch      linesearch_s;

  //time adaptation
  Vec                 Vec_uzp_time_aux, Vec_x_time_aux;

  //mech. dofs printing information
  ofstream            filg, filv;
  char                gravc[PETSC_MAX_PATH_LEN], velc[PETSC_MAX_PATH_LEN];

  // For Luzia's methods
  double h_star, Q_star, beta_l, L_min, L_max, L_range, L_low, L_sup;
  double quality_m(Mesh *mesh);
  double quality_c(Cell *cell);
  double quality_e(Cell const* cell, const int j, Mesh const* mesh);
  double sizeField(double *coords);
  PetscErrorCode meshAdapt_l();
  double TOLad;
  double quality_f(Vector a_coord, Vector b_coord);
  double sizeField_s(Vector coords);


  PetscErrorCode meshAdapt_s();
  PetscErrorCode meshFlipping_s();
  void smoothsMesh_s(Vec &Vec_normal_, Vec &Vec_x_);
  PetscErrorCode calcSlipVelocity(Vec const& Vec_x_1, Vec& Vec_slipv);
  PetscErrorCode getMeshSizes();
  PetscErrorCode orthogTest(Vec const& Vec_0, Vec const& Vec_1);
  PetscErrorCode timeAdapt();

  Vector3d Vsol, Wsol;

};

// tricks to avoid compiler error about OPENMP
#ifndef FEP_HAS_OPENMP

static int omp_get_thread_num() {return 0;};
static int omp_get_num_threads(){return 1;};

#endif
