//static char help[] = "Navier-Stokes.\n\n";

//
// ALE
//
// pho*( (Ut + (U-Umsh) · nabla)U ) + grad p = muu* div grad U + force
//
// LIMITAÇÕES:
// * triangulo e tetrahedro apenas.
// * no maximo 1 grau de liberdade associado a uma aresta
// * condição de dirichlet na normal só pode valer 0 .. procure por UNORMAL_AQUI
// * elementos isoparamétricos
//
// OBS:
// * a normal no ponto de contato é igual ao limite da normal pela superfície livre
//
//
//


#include "common.hpp"
#include <iomanip>

PetscErrorCode CheckSnesConvergence(SNES snes, PetscInt it,PetscReal xnorm, PetscReal pnorm, PetscReal fnorm, SNESConvergedReason *reason, void *ctx);
PetscErrorCode FormJacobian_mesh(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
PetscErrorCode FormFunction_mesh(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);
PetscErrorCode FormJacobian_fs(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
PetscErrorCode FormFunction_fs(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);
PetscErrorCode FormJacobian_sqrm(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
PetscErrorCode FormFunction_sqrm(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);
PetscErrorCode FormJacobian_fd(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
PetscErrorCode FormFunction_fd(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);

class AppCtx;

class Statistics;

class GetDataVelocity : public DefaultGetDataVtk
{
public:
  GetDataVelocity(double *q_array_, AppCtx const& user_) : user(user_), q_array(q_array_){}
  //double get_data_r(int nodeid) const;
  void get_vec(int id, Real * vec_out) const;
  int vec_ncomps() const { return user.mesh->spaceDim(); }
  AppCtx const& user;
  double *q_array;
  virtual ~GetDataVelocity() {}
};

class GetDataPressure : public DefaultGetDataVtk
{
public:
  GetDataPressure(double *q_array_, AppCtx const& user_) : user(user_),q_array(q_array_){}
  double get_data_r(int nodeid) const;
  AppCtx const& user;
  double *q_array;
  virtual ~GetDataPressure() {}
};

class GetDataPressCellVersion : public DefaultGetDataVtk
{
public:
  GetDataPressCellVersion(double *q_array_, AppCtx const& user_) : user(user_),q_array(q_array_){}
  double get_data_r(int cellid) const;
  AppCtx const& user;
  double *q_array;
  virtual ~GetDataPressCellVersion() {}
};

class GetDataNormal : public DefaultGetDataVtk
{
public:
  GetDataNormal(double *q_array_, AppCtx const& user_) : user(user_), q_array(q_array_){}
  void get_vec(int id, Real * vec_out) const;
  int vec_ncomps() const { return user.mesh->spaceDim(); }
  AppCtx const& user;
  double *q_array;
  virtual ~GetDataNormal() {}
};

class GetDataMeshVel : public DefaultGetDataVtk
{
public:
  GetDataMeshVel(double *q_array_, AppCtx const& user_) : user(user_), q_array(q_array_){}
  void get_vec(int id, Real * vec_out) const;
  int vec_ncomps() const { return user.mesh->spaceDim(); }
  AppCtx const& user;
  double *q_array;
  virtual ~GetDataMeshVel() {}
};

class GetDataCellTag : public DefaultGetDataVtk
{
public:
  GetDataCellTag(AppCtx const& user_) : user(user_){}
  int get_data_i(int cellid) const;
  AppCtx const& user;
  virtual ~GetDataCellTag() {}
};

class GetDataSlipVel : public DefaultGetDataVtk
{
public:
  GetDataSlipVel(double *q_array_, AppCtx const& user_) : user(user_),q_array(q_array_){}
  double get_data_r(int nodeid) const;
  AppCtx const& user;
  double *q_array;
  virtual ~GetDataSlipVel() {}
};

AppCtx::AppCtx(int argc, char **argv, bool &help_return, bool &erro)
{
  setUpDefaultOptions();
  if (getCommandLineOptions(argc, argv) )
    help_return = true;
  else
    help_return = false;

  // create some other things
  erro = createFunctionsSpace(); if (erro) return;
  createQuadrature();
}

bool AppCtx::err_checks()
{
  const bool mesh_has_edge_nodes  = mesh->numNodesPerCell() > mesh->numVerticesPerCell();
  bool const u_has_edge_assoc_dof = shape_phi_c->numDofsAssociatedToFacet() +
                                    shape_phi_c->numDofsAssociatedToCorner() > 0;
  if (!mesh_has_edge_nodes && u_has_edge_assoc_dof)
  {
    printf("ERROR: cant be superparametric element\n");
    return true;
  }
  return false;
}

void AppCtx::loadMesh()
{
  mesh.reset( Mesh::create(ECellType(mesh_cell_type),dim) );
  msh_reader.readFileMsh(filename.c_str(), mesh.get());
  vtk_printer.attachMesh(mesh.get());
  vtk_printer.isFamily(family_files);
  vtk_printer.setOutputFileName(filename_out.c_str());
  vtk_printer.setBinaryOutput(true);  //Initial true

#if false
  vtk_printer_test.attachMesh(mesh.get());
  vtk_printer_test.isFamily(family_files);
  vtk_printer_test.setOutputFileName("results/swm_tes/2d/2solDD_param_K0020n06_del02c/Z_.vtk");
  vtk_printer_test.setBinaryOutput(true);  //Initial true
#endif
  meshAliasesUpdate();
}

void AppCtx::loadDofs()
{
  dofsCreate();
  timer.restart();
  dofsUpdate();  //defines the size of the problem, i.e, n_u, n_p, n_z
  timer.elapsed("CuthillMcKeeRenumber");

  n_dofs_u_per_cell   = dof_handler[DH_UNKM].getVariable(VAR_U).numDofsPerCell();
  n_dofs_u_per_facet  = dof_handler[DH_UNKM].getVariable(VAR_U).numDofsPerFacet();
  n_dofs_u_per_corner = dof_handler[DH_UNKM].getVariable(VAR_U).numDofsPerCorner();
  n_dofs_p_per_cell   = dof_handler[DH_UNKM].getVariable(VAR_P).numDofsPerCell();
  n_dofs_p_per_facet  = dof_handler[DH_UNKM].getVariable(VAR_P).numDofsPerFacet();
  n_dofs_p_per_corner = dof_handler[DH_UNKM].getVariable(VAR_P).numDofsPerCorner();
  n_dofs_v_per_cell   = dof_handler[DH_UNKM].getVariable(VAR_M).numDofsPerCell();
  n_dofs_v_per_facet  = dof_handler[DH_UNKM].getVariable(VAR_M).numDofsPerFacet();
  n_dofs_v_per_corner = dof_handler[DH_UNKM].getVariable(VAR_M).numDofsPerCorner();
  if (is_sfip){
    n_dofs_z_per_cell   = nodes_per_cell*LZ;   //dof_handler[DH_UNKM].getVariable(VAR_Z).numDofsPerCell();
    n_dofs_z_per_facet  = nodes_per_facet*LZ;  //dof_handler[DH_UNKM].getVariable(VAR_Z).numDofsPerFacet();
    n_dofs_z_per_corner = nodes_per_corner*LZ; //dof_handler[DH_UNKM].getVariable(VAR_Z).numDofsPerCorner();
  }
  if (is_sslv){
    n_dofs_s_per_cell   = dof_handler[DH_SLIP].getVariable(VAR_S).numDofsPerCell();
    n_dofs_s_per_facet  = dof_handler[DH_SLIP].getVariable(VAR_S).numDofsPerFacet();
    n_dofs_s_per_corner = dof_handler[DH_SLIP].getVariable(VAR_S).numDofsPerCorner();
  }
}

void AppCtx::setUpDefaultOptions()
{
/* global settings */
  dim                    = 2;
  mesh_cell_type         = TRIANGLE3;
  function_space         = 1; // P1P1
  behaviors              = BH_GLS;
  //Re                   = 0.0;
  dt                     = 0.1;
  steady_tol             = 1.e-6;
  utheta                 = 1;  // time step, theta method (momentum)
  vtheta                 = 1;  // time step, theta method (mesh velocity) NOT USED
  stheta                 = 0;  // time step, theta method (mesh velocity) NOT USED
  maxts                  = 10; // max num of time steps
  finaltime              = -1;
  quadr_degree_cell      = (dim==2) ? 3 : 3;   // ordem de quadratura
  quadr_degree_facet     = 3;
  quadr_degree_corner    = 3;
  quadr_degree_err       = 8;  // to compute error
  grow_factor            = 0.05;
  print_step             = 1;
  PI                     = 1;
  PIs                    = 1;
  temporal_solver        = 0;  //to choose a temporal solver, 0 for mr, stokes
  n_unknowns_ups         = 0;
  n_modes                = 0;
  n_links                = 0;
  n_groups               = 0;
  nforp                  = 0;
  Kforp                  = 0.0;
  Kcte                   = 0.0;
  Kelast                 = 0.0;
  family_files           = PETSC_TRUE;
  has_convec             = PETSC_TRUE;
  renumber_dofs          = PETSC_FALSE;
  fprint_ca              = PETSC_FALSE;
  nonlinear_elasticity   = PETSC_FALSE;
  mesh_adapt             = PETSC_TRUE;
  static_mesh            = PETSC_FALSE;
  time_adapt             = PETSC_TRUE;
  fprint_hgv             = PETSC_FALSE;
  is_sslv                = PETSC_FALSE;
  is_sfim                = PETSC_FALSE;
  is_sflp                = PETSC_FALSE;
  is_curvt               = PETSC_FALSE;
  is_unksv               = PETSC_FALSE;
  is_axis                = PETSC_FALSE;
  exact_normal           = PETSC_FALSE;
  force_pressure         = PETSC_FALSE;  // elim null space (auto)
  print_to_matlab        = PETSC_FALSE;  // imprime o jacobiano e a função no formato do matlab
  force_dirichlet        = PETSC_TRUE;   // impõe cc ? (para debug)
  full_diriclet          = PETSC_TRUE;
  force_mesh_velocity    = PETSC_FALSE;
  plot_exact_sol         = PETSC_FALSE;
  unsteady               = PETSC_TRUE;
  boundary_smoothing     = PETSC_TRUE;
  dup_press_nod          = PETSC_FALSE;
  s_dofs_elim            = PETSC_FALSE;
  inverted_elem          = PETSC_FALSE;

  is_mr_ab           = PETSC_FALSE;
  is_bdf3            = PETSC_FALSE;
  is_bdf2            = PETSC_FALSE;
  is_bdf2_bdfe       = PETSC_FALSE;
  is_bdf2_ab         = PETSC_FALSE;
  is_bdf_cte_vel     = PETSC_FALSE;
  is_bdf_euler_start = PETSC_FALSE;
  is_bdf_extrap_cte  = PETSC_FALSE;
  is_basic           = PETSC_FALSE;

  is_Stokes          = false;
  is_NavierStokes    = false;

  solve_the_sys          = true;   // for debug
  filename               = (dim==2 ? "malha/cavity2d-1o.msh" : "malha/cavity3d-1o.msh");

  //Luzia's part for meshAdapt_l(), meshAdapt_s() based on the fields of distances
  //For swimmers of size 220um, eg, opalina of semiaxis 110um
/*  h_star = 0.02;
  Q_star = 0.9;//0.6;
  beta_l = 4.00;
  L_min  = 3.7113111772657698672617243573768;//1.0;//0.3; //0.02;//h_star*sqrt(3)/3.0;
  L_max  = 639.78926362886295464704744517803; //sqrt(3);// 1;
  L_range = 200.0;//0.3;
  L_low = 0.1; //0.1;//L_min/L_max;// + 0.01; //0.577350269;//1.0*L_min/L_max;
  L_sup = 10.0;//1.732050808; //L_max/L_min - 0.01; //1.732050808; //1.0/L_low;
*/
  h_star = 0.02;
  Q_star = 0.8;
  beta_l = 4.00;
  L_min  = 0.3; //0.02;//h_star*sqrt(3)/3.0;
  L_max = 1;
  L_range = 0.3;
  L_low = 0.577350269;//1.0*L_min/L_max;
  L_sup = 1.732050808; //1.0/L_low;

  TOLad = 0.7;  //for meshAdapt_d() based on the mean of the neighboor edges length

  n_unknowns_z      = 0; n_unknowns_f        = 0; n_unknowns_s        = 0;
  n_nodes_fsi       = 0; n_nodes_fo          = 0; n_nodes_so          = 0;
  n_dofs_z_per_cell = 0; n_dofs_z_per_facet  = 0; n_dofs_z_per_corner = 0;
  n_dofs_s_per_cell = 1; n_dofs_s_per_facet  = 1; n_dofs_s_per_corner = 1;
}

bool AppCtx::getCommandLineOptions(int argc, char **/*argv*/)
{
  PetscBool          flg_fin, flg_fout, flg_min, flg_hout, flg_sin, flg_iin;
  char               finaux[PETSC_MAX_PATH_LEN], minaux[PETSC_MAX_PATH_LEN], sinaux[PETSC_MAX_PATH_LEN];
  char               foutaux[PETSC_MAX_PATH_LEN], houtaux[PETSC_MAX_PATH_LEN], iinaux[PETSC_MAX_PATH_LEN];
  PetscBool          ask_help;

  if (argc == 1)
  {
    cout << "\nusage:\n";
    cout << "\n\t./main `cat args`\n\n";
    cout << "where args is a file with the command line parameters, or\n\n";
    cout << "\t./main -help\n\n";
    cout << "to show options.\n\n";
    return true;
  }

  /* opções do usuário */ //*Esto se imprime cuando se da -help
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the Navier-Stokes", "ahn?");   //Mandatory to PetscOptions*
  PetscOptionsInt("-dim", "space dimension", "main.cpp", dim, &dim, PETSC_NULL);
  PetscOptionsInt("-mesh_type", "mesh type", "main.cpp", mesh_cell_type, &mesh_cell_type, PETSC_NULL);
  PetscOptionsInt("-function_space", "function_space", "main.cpp", 1, &function_space, PETSC_NULL);
  PetscOptionsInt("-maxts", "maximum number of time steps", "main.cpp", maxts, &maxts, PETSC_NULL);
  PetscOptionsInt("-quadr_e", "quadrature degree (for calculating the error)", "main.cpp", quadr_degree_err, &quadr_degree_err, PETSC_NULL);
  PetscOptionsInt("-quadr_c", "quadrature degree", "main.cpp", quadr_degree_cell, &quadr_degree_cell, PETSC_NULL);
  PetscOptionsInt("-quadr_f", "quadrature degree (facet)", "main.cpp", quadr_degree_facet, &quadr_degree_facet, PETSC_NULL);
  PetscOptionsInt("-quadr_r", "quadrature degree (corner)", "main.cpp", quadr_degree_corner, &quadr_degree_corner, PETSC_NULL);
  PetscOptionsInt("-print_step", "print_step", "main.cpp", print_step, &print_step, PETSC_NULL);
  PetscOptionsInt("-picard_iter_init_cond", "picard iteration for initial conditions", "main.cpp", PI, &PI, PETSC_NULL);
  PetscOptionsInt("-picard_iter_solver", "picard iteration for solver", "main.cpp", PIs, &PIs, PETSC_NULL);
  PetscOptionsInt("-temporal_solver", "choose time solver", "main.cpp", temporal_solver, &temporal_solver, PETSC_NULL);
  PetscOptionsInt("-stabilization_method", "choose stabilization method", "main.cpp", stabilization_method, &stabilization_method, PETSC_NULL);
  PetscOptionsInt("-n_modes", "number of elastic modes", "main.cpp", n_modes, &n_modes, PETSC_NULL);
  PetscOptionsInt("-n_links", "number of links in swimmer", "main.cpp", n_links, &n_links, PETSC_NULL);
  PetscOptionsInt("-n_groups", "number of groups swimmers", "main.cpp", n_groups, &n_groups, PETSC_NULL);
  PetscOptionsInt("-theta_dof_version", "solve the 2D problem with the angle dof", "main.cpp", thetaDOF, &thetaDOF, PETSC_NULL);
//PetscOptionsScalar("-Re", "Reynolds number", "main.cpp", Re, &Re, PETSC_NULL);
  PetscOptionsScalar("-dt", "time step", "main.cpp", dt, &dt, PETSC_NULL);
  PetscOptionsScalar("-utheta", "utheta value", "main.cpp", utheta, &utheta, PETSC_NULL);
  PetscOptionsScalar("-vtheta", "vtheta value", "main.cpp", vtheta, &vtheta, PETSC_NULL);  //borrar
  PetscOptionsScalar("-stheta", "stheta value", "main.cpp", stheta, &stheta, PETSC_NULL);  //borrar
  PetscOptionsScalar("-sst", "steady state tolerance", "main.cpp", steady_tol, &steady_tol, PETSC_NULL);
  PetscOptionsScalar("-beta1", "par vel do fluido", "main.cpp", beta1, &beta1, PETSC_NULL);
  PetscOptionsScalar("-beta2", "par vel elastica", "main.cpp", beta2, &beta2, PETSC_NULL);
  PetscOptionsScalar("-finaltime", "the simulation ends at this time.", "main.cpp", finaltime, &finaltime, PETSC_NULL);
  PetscOptionsScalar("-Kforp", "wave constant paramecium test", "main.cpp", Kforp, &Kforp, PETSC_NULL);
  PetscOptionsScalar("-nforp", "wave number paramecium test", "main.cpp", nforp, &nforp, PETSC_NULL);
  PetscOptionsScalar("-Kcte", "transition constant for linear tangent force", "main.cpp", Kcte, &Kcte, PETSC_NULL);
  PetscOptionsScalar("-Kelast", "transition constant for linear tangent force", "main.cpp", Kelast, &Kelast, PETSC_NULL);
  PetscOptionsBool("-print_to_matlab", "print jacobian to matlab", "main.cpp", print_to_matlab, &print_to_matlab, PETSC_NULL);
  PetscOptionsBool("-force_dirichlet", "force dirichlet bound cond", "main.cpp", force_dirichlet, &force_dirichlet, PETSC_NULL);
  PetscOptionsBool("-force_pressure", "force pressure", "main.cpp", force_pressure, &force_pressure, PETSC_NULL);
  PetscOptionsBool("-duplicate_pressure_nodes", "duplicate pressure nodes", "main.cpp", dup_press_nod, &dup_press_nod, PETSC_NULL);
  PetscOptionsBool("-plot_es", "plot exact solution", "main.cpp", plot_exact_sol, &plot_exact_sol, PETSC_NULL);
  PetscOptionsBool("-family_files", "plot family output", "main.cpp", family_files, &family_files, PETSC_NULL);
  PetscOptionsBool("-has_convec", "convective term", "main.cpp", has_convec, &has_convec, PETSC_NULL);
  PetscOptionsBool("-unsteady", "unsteady problem", "main.cpp", unsteady, &unsteady, PETSC_NULL);
  PetscOptionsBool("-boundary_smoothing", "boundary_smoothing", "main.cpp", boundary_smoothing, &boundary_smoothing, PETSC_NULL);
  PetscOptionsBool("-force_mesh_velocity", "force_mesh_velocity", "main.cpp", force_mesh_velocity, &force_mesh_velocity, PETSC_NULL);
  PetscOptionsBool("-solve_slip_velocity", "activate slip velocity solver", "main.cpp", is_sslv, &is_sslv, PETSC_NULL);
  PetscOptionsBool("-solve_axisymmetric", "activate axisymmetric solver", "main.cpp", is_axis, &is_axis, PETSC_NULL);
  PetscOptionsBool("-renumber_dofs", "renumber dofs", "main.cpp", renumber_dofs, &renumber_dofs, PETSC_NULL);
  PetscOptionsBool("-fprint_ca", "print contact angle", "main.cpp", fprint_ca, &fprint_ca, PETSC_NULL);
  PetscOptionsBool("-nonlinear_elasticity", "put a non-linear term in the elasticity problem", "main.cpp", nonlinear_elasticity, &nonlinear_elasticity, PETSC_NULL);
  PetscOptionsBool("-mesh_adapt", "adapt the mesh during simulation", "main.cpp", mesh_adapt, &mesh_adapt, PETSC_NULL);
  PetscOptionsBool("-static_mesh", "have a static mesh during simulation", "main.cpp", static_mesh, &static_mesh, PETSC_NULL);
  PetscOptionsBool("-time_adapt", "adapt the time during simulation", "main.cpp", time_adapt, &time_adapt, PETSC_NULL);
  PetscOptionsBool("-curved_trian", "enable curved elements", "main.cpp", is_curvt, &is_curvt, PETSC_NULL);
  PetscOptionsBool("-solve_coupled_slip_velocity", "solve the slip velocity coupled with the UPS problem", "main.cpp", is_unksv, &is_unksv, PETSC_NULL);
  PetscOptionsBool("-use_exact_normal", "use exact solid normal", "main.cpp", exact_normal, &exact_normal, PETSC_NULL);
  PetscOptionsBool("-solid_dofs_elimination", "eliminate solid dofs controlled by a condition function", "main.cpp", s_dofs_elim, &s_dofs_elim, PETSC_NULL);
  PetscOptionsBool("-ale", "mesh movement", "main.cpp", ale, &ale, PETSC_NULL);
  PetscOptionsGetString(PETSC_NULL,"-fin",finaux,PETSC_MAX_PATH_LEN-1,&flg_fin);
  PetscOptionsGetString(PETSC_NULL,"-fout",foutaux,PETSC_MAX_PATH_LEN-1,&flg_fout);
  PetscOptionsGetString(PETSC_NULL,"-min",minaux,PETSC_MAX_PATH_LEN-1,&flg_min);
  PetscOptionsGetString(PETSC_NULL,"-hout",houtaux,PETSC_MAX_PATH_LEN-1,&flg_hout);
  PetscOptionsGetString(PETSC_NULL,"-sin",sinaux,PETSC_MAX_PATH_LEN-1,&flg_sin);
  PetscOptionsGetString(PETSC_NULL,"-iin",iinaux,PETSC_MAX_PATH_LEN-1,&flg_iin);
  PetscOptionsHasName(PETSC_NULL,"-help",&ask_help);

  switch(temporal_solver)
  {
    case 0:{is_mr_qextrap       = PETSC_TRUE; break;}
    case 1:{is_mr_ab            = PETSC_TRUE; break;}
    case 2:{is_bdf3             = PETSC_TRUE; break;}
    case 3:{is_bdf2             = PETSC_TRUE; break;}
    case 4:{is_bdf2_bdfe        = PETSC_TRUE; break;}
    case 5:{is_bdf2_ab          = PETSC_TRUE; break;}
    case 6:{is_bdf_cte_vel      = PETSC_TRUE; break;}
    case 7:{is_bdf_euler_start  = PETSC_TRUE; break;}
    case 8:{is_bdf_extrap_cte   = PETSC_TRUE; break;}
    case 9:{is_basic            = PETSC_TRUE; break;}
  }

  if ((is_bdf2 && utheta!=1) || (is_bdf3 && utheta!=1))
  {
    cout << "ERROR: BDF2/3 with utheta!=1" << endl;
    throw;
  }
  if (is_bdf_extrap_cte && !is_bdf_cte_vel)
  {
    cout << "ERROR: is_bdf_extrap_cte && !is_bdf_cte_vel" << endl;
    throw;
  }
  if (is_bdf2_bdfe && !is_bdf2)
  {
    cout << "ERROR: !is_bdf_bdf_extrap && !is_bdf2" << endl;
    throw;
  }
  if (is_bdf2_ab && !is_bdf2)
  {
    cout << "ERROR: !is_bdf_ab && !is_bdf2" << endl;
    throw;
  }
  if (is_bdf3 && is_bdf2)
  {
    cout << "ERROR: is_bdf3 && is_bdf2" << endl;
    throw;
  }


  if (finaltime < 0)
    finaltime = maxts*dt;
  else
    maxts = 1 + static_cast<int> (round ( finaltime/dt ));

  dirichlet_tags.resize(16);
  PetscBool flg_tags;
  int nmax = dirichlet_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-dir_tags", dirichlet_tags.data(), &nmax, &flg_tags); //looks for -dir_tags and find nmax and gives bool value to flg_tags
  if (flg_tags)
    dirichlet_tags.resize(nmax);
  else
    dirichlet_tags.clear();

  neumann_tags.resize(16);
  nmax = neumann_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-neum_tags", neumann_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    neumann_tags.resize(nmax);
  else
    neumann_tags.clear();

  interface_tags.resize(16);
  nmax = interface_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-interf_tags", interface_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    interface_tags.resize(nmax);
  else
    interface_tags.clear();

  solid_tags.resize(16);
  nmax = solid_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-solid_tags", solid_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    solid_tags.resize(nmax);
  else
    solid_tags.clear();

  triple_tags.resize(16);
  nmax = triple_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-triple_tags", triple_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    triple_tags.resize(nmax);
  else
    triple_tags.clear();

  periodic_tags.resize(16);
  nmax = periodic_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-periodic_tags", periodic_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    periodic_tags.resize(nmax);
  else
    periodic_tags.clear();
  if (periodic_tags.size() % 2 != 0)
  {
    std::cout << "Invalid periodic tags\n";
    throw;
  }

  feature_tags.resize(16);
  nmax = feature_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-feature_tags", feature_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    feature_tags.resize(nmax);
  else
    feature_tags.clear();

  fluidonly_tags.resize(16);  //cout << flusol_tags.max_size() << endl;
  nmax = fluidonly_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-fonly_tags", fluidonly_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
	fluidonly_tags.resize(nmax);
  else
	fluidonly_tags.clear();

  solidonly_tags.resize(1e8);
  nmax = solidonly_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-sonly_tags", solidonly_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    {solidonly_tags.resize(nmax); is_sfip = PETSC_TRUE; is_slipv = PETSC_TRUE;}
  else
    {solidonly_tags.clear(); is_sfip = PETSC_FALSE; is_slipv = PETSC_FALSE;}
  n_solids = solidonly_tags.size();
  LZ = dim*(dim+1)/2 + n_modes; //3*(dim-1);

  flusoli_tags.resize(1e8);  //cout << flusol_tags.max_size() << endl;
  nmax = flusoli_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-fsi_tags", flusoli_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    flusoli_tags.resize(nmax);
  else
    flusoli_tags.clear();

  slipvel_tags.resize(1e8);
  nmax = slipvel_tags.size();
  PetscOptionsGetIntArray(PETSC_NULL, "-slipv_tags", slipvel_tags.data(), &nmax, &flg_tags);
  if (flg_tags)
    {slipvel_tags.resize(nmax); /*is_slipv = PETSC_TRUE;*/}
  else
    {slipvel_tags.clear(); /*is_slipv = PETSC_FALSE;*/}

  PetscOptionsEnd();   //Finish PetscOptions*

  switch (function_space)
  {
    case P1P1:
    {
      if(dim==2) mesh_cell_type = TRIANGLE3;
      else       mesh_cell_type = TETRAHEDRON4;
      break;
    }
    case P1bP1_c:
    {
      if(dim==2) mesh_cell_type = TRIANGLE3;
      else       mesh_cell_type = TETRAHEDRON4;
      break;
    }
    case P2bPm1_c:
    {
      if(dim==2) mesh_cell_type = TRIANGLE6;
      else       mesh_cell_type = TETRAHEDRON10;
      break;
    }
    case P2P1:
    {
      if(dim==2) mesh_cell_type = TRIANGLE6;
      else       mesh_cell_type = TETRAHEDRON10;
      break;
    }
    case P1bP1:
    {
      if(dim==2) mesh_cell_type = TRIANGLE3;
      else       mesh_cell_type = TETRAHEDRON4;
      break;
    }
    case P2P0:
    {
      if(dim==2) mesh_cell_type = TRIANGLE6;
      else       mesh_cell_type = TETRAHEDRON10;
      break;
    }
    case P2bPm1:
    {
      if(dim==2) mesh_cell_type = TRIANGLE6;
      else       mesh_cell_type = TETRAHEDRON10;
      break;
    }
    case P1P1unstable:
    {
      if(dim==2) mesh_cell_type = TRIANGLE3;
      else       mesh_cell_type = TETRAHEDRON4;
      break;
    }
    case P2bP1:
    {
      if(dim==2) mesh_cell_type = TRIANGLE6;
      else       mesh_cell_type = TETRAHEDRON10;
      break;
    }
    default:
    {
      cout << "invalid function space " << function_space << endl;
      throw;
    }
  };


  if(ask_help)
    return true;

  if (flg_fin)
    filename.assign(finaux);

  if (flg_fout)
    filename_out.assign(foutaux);

  if (flg_min){
    filemass.assign(minaux);
    int N = n_solids;
    double mass = 0.0, rad1 = 0.0, rad2 = 0.0, rad3 = 0.0, vol = 0.0, the = 0.0,
           xg = 0.0, yg = 0.0, zg = 0.0, llin = 0.0, mode = 0.0;
    Vector3d Xg, rad;
    ifstream is;
    is.open(filemass.c_str(),ifstream::in);
    if (!is.good()) {cout << "mass file not found" << endl; throw;}
    else {
      cout << "mass file format: mass volume radius(ellipsoidal case 2D or 3D) init.angle init.center";
      if (n_modes > 0) {cout << " modes";}
      if (n_links > 0) {cout << " max_link_length";}
      cout << endl;
    }
    for (; N > 0; N--){
      is >> mass;
      is >> vol;
      is >> rad1; is >> rad2; if(dim == 3){is >> rad3;}
      is >> the;
      is >> xg; is >> yg; if(dim == 3){is >> zg;}
      if(n_modes > 0){
        for (int nm = 0; nm < n_modes; nm++){is >> mode;}//TODO: modes > 1
      }
      if(n_links > 0){is >> llin;}
      rad << rad1, rad2, rad3;
      Xg << xg, yg, zg;
      MV.push_back(mass); RV.push_back(rad); VV.push_back(vol);
      XG_0.push_back(Xg);      XG_1.push_back(Xg);      XG_m1.push_back(Xg);      XG_ini.push_back(Xg);
      theta_0.push_back(the);  theta_1.push_back(the);  theta_m1.push_back(the);  theta_ini.push_back(the); //(rand()%6);
      llink_0.push_back(llin); llink_1.push_back(llin); llink_m1.push_back(llin); llink_ini.push_back(llin); //(rand()%6);
      modes_0.push_back(mode); modes_1.push_back(mode); modes_m1.push_back(mode); modes_ini.push_back(mode); //(rand()%6);
    }
    is.close();
    MV.resize(n_solids); RV.resize(n_solids); VV.resize(n_solids);
    XG_0.resize(n_solids);    XG_1.resize(n_solids);    XG_m1.resize(n_solids);    XG_ini.resize(n_solids);
    theta_0.resize(n_solids); theta_1.resize(n_solids); theta_m1.resize(n_solids); theta_ini.resize(n_solids);
    llink_0.resize(n_links);  llink_1.resize(n_links);  llink_m1.resize(n_links);  llink_ini.resize(n_links);
    modes_0.resize(n_modes);  modes_1.resize(n_modes);  modes_m1.resize(n_modes);  modes_ini.resize(n_modes);
  }

  if (flg_sin){
    filexas.assign(sinaux);
    int N = 702;
    double rad = 0.0, uu = 0.0;
    ifstream is;
    is.open(filexas.c_str(),ifstream::in);
    if (!is.good()) {cout << "exact sol file not found" << endl; throw;}
    for (; N > 0; N--){
      is >> rad; is >> uu; //cout << rad << "   " << uu << endl;
      RR.push_back(rad); UU.push_back(uu);
    }
    is.close();
    RR.resize(N); UU.resize(N);
  }

  if (flg_hout){
    filehist_out.assign(houtaux);
    fprint_hgv = PETSC_TRUE;
    struct stat st;
    if (stat(filehist_out.c_str(), &st) == -1){
      mkdir(filehist_out.c_str(), 0770);
    }
    if (print_to_matlab){
      char matrices_out[PETSC_MAX_PATH_LEN];
      sprintf(matrices_out,"%s/matrices",filehist_out.c_str());
      if (stat(matrices_out, &st) == -1){
        mkdir(matrices_out, 0770);
      }
    }
  }

  if (flg_iin){
    read_from_sv_fd = PETSC_TRUE;
    filesvfd.assign(iinaux);
  }
  else
    read_from_sv_fd = PETSC_FALSE;

  //if (neumann_tags.size() + interface_tags.size() == 0 || force_pressure)
  if (force_pressure)
  {
    full_diriclet = PETSC_TRUE;
    force_pressure = PETSC_TRUE;
  }
  else
    force_pressure = PETSC_FALSE;

  if (!unsteady)
  {
    if (ale) {
      printf("ALE with steady problem?\n");
      //throw;
    }

    if (utheta != 1)
    {
      cout << "WARNING!!!\n";
      cout << ">>> Steady problem... setting utheta to 1" << endl;
      utheta = 1;
    }

  }

  if ((is_slipv && !is_sfip) || (is_slipv && !is_sfip))
  {
    printf("To solve the slip velocity problem you need to set a SFI problem, check args\n");
    throw;
  }

  if (n_modes > 0){
    is_sfim = PETSC_TRUE;
  }

  if (n_links > 0){
    is_sflp = PETSC_TRUE;
    if (n_groups > 0){
      //ebref.resize(n_links);
    }
  }

  if (dim == 2){
    if (quadr_degree_err > 10){
      cout << ">>> Exceeded maximum quadrature error order, setting it to 10" << endl;
      quadr_degree_err = 10;
    }
  }
  else if (dim == 3){
    if (quadr_degree_err > 8){
      cout << ">>> Exceeded maximum quadrature error order, setting it to 8" << endl;
      quadr_degree_err = 8;
    }
  }

  switch (stabilization_method)
  {// st_met = -1 for Douglas and Wang, +1 for Hughes and Franca, 0 for neither
    case 0:{st_met =  0; st_vis = 0; break;}
    case 1:{st_met =  1; st_vis = 0; break;}
    case 2:{st_met = -1; st_vis = 0; break;}
    case 3:{st_met =  0; st_vis = 1; break;}
    case 4:{st_met =  1; st_vis = 1; break;}
    case 5:{st_met = -1; st_vis = 1; break;}
  }//st_vis = 1 for including visc term in the residual, 0 for not including

  return false;
}

bool AppCtx::createFunctionsSpace()
{
  EShapeType velo_shape, pres_shape;
  bool is_simplex = ctype2cfamily(ECellType(mesh_cell_type)) == SIMPLEX;
  ECellType cell_type = ECellType(mesh_cell_type);

  switch (function_space)
  {
    case 1: // P1P1 (or Q1Q1) GLS stabilization
      {
        behaviors = BH_GLS;
        velo_shape = is_simplex ? P1 : Q1;
        pres_shape = is_simplex ? P1 : Q1;
      }
      break;
    case 2: // P1+P1 with bubble condensation
      {
        behaviors = BH_bble_condens_PnPn;
        velo_shape = P1;
        pres_shape = P1;
      }
      break;
    case 3: // P2+Pm1 with bubble condensation and pressure gradient elimination
      {
        behaviors = BH_bble_condens_CR;
        velo_shape = P2;
        pres_shape = P0;
      }
      break;
    case 4: // P2P1 (or Q2Q1)
      {
        behaviors = 0;
        velo_shape = is_simplex ? P2 : Q2;
        pres_shape = is_simplex ? P1 : Q1;
      }
      break;
    case 5: // P1+P1 traditional
      {
        behaviors = 0;
        velo_shape = P1ph ;
        pres_shape = P1;
      }
      break;
    case 6: // P2P0
      {
        behaviors = 0;
        velo_shape = P2;
        pres_shape = P0;
      }
      break;
    case 7: // P2+Pm1 full
      {
        behaviors = 0;
        velo_shape = P2ph;
        pres_shape = Pm1;
      }
      break;
    case 8: // P1P1 unstable
      {
        behaviors = 0;
        velo_shape = P1;
        pres_shape = P1;
      }
      break;
    case 9: // P2bP1 bubble condensation
      {
        behaviors = BH_bble_condens_PnPn;
        velo_shape = is_simplex ? P2 : Q2;
        pres_shape = is_simplex ? P1 : Q1;
      }
      break;
    default:
      {
        behaviors = BH_GLS;
        velo_shape = is_simplex ? P1 : Q1;
        pres_shape = is_simplex ? P1 : Q1;
      }

  }

  shape_phi_c.reset(ShapeFunction::create(cell_type, velo_shape));
  shape_psi_c.reset(ShapeFunction::create(cell_type, pres_shape));
  shape_qsi_c.reset(ShapeFunction::create(cell_type));
  shape_phi_f.reset(ShapeFunction::create(facetof(cell_type), facetof(velo_shape)));
  shape_psi_f.reset(ShapeFunction::create(facetof(cell_type), facetof(pres_shape)));
  shape_qsi_f.reset(ShapeFunction::create(facetof(cell_type)));
  shape_phi_r.reset(ShapeFunction::create(facetof(facetof(cell_type)), facetof(facetof(velo_shape))));
  shape_psi_r.reset(ShapeFunction::create(facetof(facetof(cell_type)), facetof(facetof(pres_shape))));
  shape_qsi_r.reset(ShapeFunction::create(facetof(facetof(cell_type))));

  shape_bble.reset(ShapeFunction::create(cell_type, BUBBLE));

  pres_pres_block = false;
  if (behaviors & (BH_bble_condens_PnPn | BH_GLS))
    pres_pres_block = true;

  return false;
}

void AppCtx::createQuadrature()
{
  ECellType ct = ECellType(mesh_cell_type);

  quadr_cell.reset( Quadrature::create(ECellType(ct)) );
  quadr_cell->setOrder(quadr_degree_cell);
  n_qpts_cell = quadr_cell->numPoints();

  quadr_facet.reset( Quadrature::create(facetof(ECellType(ct))) );
  quadr_facet->setOrder(quadr_degree_facet);
  n_qpts_facet = quadr_facet->numPoints();

  quadr_corner.reset( Quadrature::create(facetof(facetof(ECellType(ct)))) );
  quadr_corner->setOrder(quadr_degree_corner);
  n_qpts_corner = quadr_corner->numPoints();

  quadr_err.reset( Quadrature::create(ECellType(ct)) );
  quadr_err->setOrder(quadr_degree_err);
  n_qpts_err = quadr_err->numPoints();

}

void AppCtx::meshAliasesUpdate()
{
  n_nodes  = mesh->numNodes();
  n_cells  = mesh->numCells();
  n_facets = mesh->numFacets();
  n_corners= mesh->numCorners();
  n_nodes_total  = mesh->numNodesTotal();   // includes disables
  n_cells_total  = mesh->numCellsTotal();   // includes disables
  n_facets_total = mesh->numFacetsTotal();  // includes disables
  n_corners_total= mesh->numCornersTotal(); // includes disables
  nodes_per_cell  = mesh->numNodesPerCell();
  nodes_per_facet = mesh->numNodesPerFacet();
  nodes_per_corner= mesh->numNodesPerCorner();
}

void AppCtx::dofsCreate()
{
  // mesh velocity
  dof_handler[DH_MESH].setMesh(mesh.get());
  dof_handler[DH_MESH].addVariable("velo_mesh",  shape_qsi_c.get(), dim);
  //dof_handler[DH_MESH].setVariablesRelationship(blocks.data());

//  int const * fo_tag = &fluidonly_tags[0];
  dof_handler[DH_UNKM].setMesh(mesh.get());
//    dof_handler[DH_UNKM].addVariable("velo_fluid", shape_phi_c.get(), dim, fluidonly_tags.size(), fo_tag);
  dof_handler[DH_UNKM].addVariable("velo_fluid", shape_phi_c.get(), dim);
  dof_handler[DH_UNKM].addVariable("pres_fluid", shape_psi_c.get(), 1);
  if (dup_press_nod)
    dof_handler[DH_UNKM].getVariable(VAR_P).setType(SPLITTED_BY_REGION_CELL,0,0);

  if (is_sfip && false){ //the varible "velo_solid" gives problems with getSparsityTable()
    int const * fsi_tag = &flusoli_tags[0];
    dof_handler[DH_UNKM].addVariable("velo_solid", shape_phi_c.get(), LZ, flusoli_tags.size(), fsi_tag);
//    int const * fo_tag = &fluidonly_tags[0];
//    int const * so_tag = &solidonly_tags[0];
//    dof_handler[DH_FOON].setMesh(mesh.get());
//    dof_handler[DH_FOON].addVariable("vel_ofluid", shape_phi_c.get(), dim, fluidonly_tags.size(), fo_tag);
//    dof_handler[DH_FOON].addVariable("vel_osolid", shape_phi_c.get(), dim, fluidonly_tags.size(), so_tag);
  }
  if (is_sslv){
    // slip velocity
    dof_handler[DH_SLIP].setMesh(mesh.get());
    dof_handler[DH_SLIP].addVariable("velo_slip",  shape_qsi_c.get(), 1); //shape_psi_c
  }

}

bool isPeriodic(Point const* p1, Point const* p2, int dim)
{
  double const TOL = 1.e-9;

  if (dim == 2)
  {
    if (fabs(p1->getCoord(0) - p2->getCoord(0)) < TOL)
      return true;
    if (fabs(p1->getCoord(1) - p2->getCoord(1)) < TOL)
      return true;
  }
  else // dim == 3
  {
    int c = 0;
    if (fabs(p1->getCoord(0) - p2->getCoord(0)) < TOL)
      c++;
    if (fabs(p1->getCoord(1) - p2->getCoord(1)) < TOL)
      c++;
    if (fabs(p1->getCoord(2) - p2->getCoord(2)) < TOL)
      c++;

    if (c==2)
      return true;
  }

  return false;
}

void AppCtx::dofsUpdate()
{
  dof_handler[DH_UNKM].SetUp();
  if (renumber_dofs)
    dof_handler[DH_UNKM].CuthillMcKeeRenumber();
  n_nodes_fsi = 0; n_nodes_fo = 0; n_nodes_so = 0; n_nodes_sv = 0; n_nodes_fs = 0; n_facets_fsi = 0;
  std::vector<int> dofs1;
  std::vector<int> dofs2;
  NN_Solids.assign(n_solids,0);
  int tag, nod_id;

  int dofs_a[10];
  int dofs_b[10];

  // apply periodic boundary conditions here //////////////////////////////////////////////////
  {
    if ((mesh_cell_type != TRIANGLE3) && (mesh_cell_type != TETRAHEDRON4) && !periodic_tags.empty() )
    {
      std::cout << "Periodic boundary conditions is not allowed with high order mesh\n";
      throw;
    }

      point_iterator point1 = mesh->pointBegin();
      point_iterator point1_end = mesh->pointEnd();
      point_iterator point2, point2_end;

      for (; point1 != point1_end; ++point1)
      {
    	tag = point1->getTag();
        nod_id = is_in_id(tag,flusoli_tags);
        if (nod_id) {NN_Solids[nod_id-1]++; n_nodes_fsi++;}
        if (is_in(tag,interface_tags)) {n_nodes_fsi++;}
    	if (is_in(tag,fluidonly_tags)) {n_nodes_fo++;}
    	if (is_in(tag,solidonly_tags)) {n_nodes_so++;}
        if (is_in(tag,slipvel_tags)) {n_nodes_sv++;}
        if (is_in(tag,flusoli_tags)) {n_nodes_fs++;}

        for (int i = 0; i < (int)periodic_tags.size(); i+=2)
        {

          int const tag1 = periodic_tags[i];
          int const tag2 = periodic_tags[i+1];

          if (tag1 != point1->getTag())
            continue;

          point2 = mesh->pointBegin();
          point2_end = mesh->pointEnd();
          for (; point2 != point2_end; ++point2)
          {
            if (tag2 != point2->getTag())
              continue;

            if (isPeriodic(&*point1, &*point2, dim))
            {
              for (int var = 0; var < 2; ++var)
              {
                int ndpv = dof_handler[DH_UNKM].getVariable(var).numDofsPerVertex();

                dof_handler[DH_UNKM].getVariable(var).getVertexDofs(dofs_a, &*point1);
                dof_handler[DH_UNKM].getVariable(var).getVertexDofs(dofs_b, &*point2);
                for (int k = 0; k < ndpv; ++k)
                {
                  dofs1.push_back(dofs_a[k]);
                  dofs2.push_back(dofs_b[k]);
                }
              }

            }
          }
        }
      }
  }////////////////////////////////////////////////////////////////////////////////////////////////////
//  dof_handler[DH_UNKS].linkDofs(dofs1.size(), dofs1.data(), dofs2.data());
//  n_unknowns = dof_handler[DH_UNKS].numDofs();

  dof_handler[DH_MESH].SetUp();
  n_dofs_v_mesh = dof_handler[DH_MESH].numDofs();

  dof_handler[DH_UNKM].linkDofs(dofs1.size(), dofs1.data(), dofs2.data());
  n_unknowns_u = dof_handler[DH_UNKM].getVariable(VAR_U).numPositiveDofs();
  n_unknowns_p = dof_handler[DH_UNKM].getVariable(VAR_P).numPositiveDofs();
  n_unknowns_z = n_solids*LZ;
  //cout << n_unknowns_z << "   " <<  dof_handler[DH_UNKM].getVariable(VAR_Z).numPositiveDofs();

//    dof_handler[DH_FOON].SetUp();
//    n_unknowns_f = dof_handler[DH_FOON].getVariable(VAR_F).numPositiveDofs();
//    n_unknowns_s = dof_handler[DH_FOON].getVariable(VAR_S).numPositiveDofs();

  n_unknowns_fs = n_unknowns_u + n_unknowns_p + n_unknowns_z + n_links; //- dim*n_nodes_fsi;
  //if (is_unksv){
  //  n_unknowns_ups = n_unknowns_fs;
  //  n_unknowns_fs += n_unknowns_u;
  //}

  //for (unsigned int l = 0; l < NN_Solids.size(); l++) cout << NN_Solids[l] << " ";
  if (is_sslv){
    dof_handler[DH_SLIP].SetUp();
    n_unknowns_sv = dof_handler[DH_SLIP].getVariable(VAR_S).numPositiveDofs();
  }

  //surface facets counting//////////////////////////////////////////////////
  facet_iterator facet = mesh->facetBegin();
  facet_iterator facet_end = mesh->facetEnd();
  for (;facet != facet_end; ++facet)
  {
    int tag_e = facet->getTag();
    if (is_in(tag_e,slipvel_tags)||is_in(tag_e,flusoli_tags)){n_facets_fsi++;}
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////
}

PetscErrorCode AppCtx::allocPetscObjs()
{
  printf("Allocating PETSC objects... ");

  //PetscErrorCode      ierr;
  PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INDEX);

  allocPetscObjsVecInt();
  allocPetscObjsVecNoInt();
  allocPetscObjsMesh();
  allocPetscObjsFluid();

//  std::vector<int> nnz;
/*
  //  ------------------------------------------------------------------
  //  ----------------------- Mesh -------------------------------------
  //  ------------------------------------------------------------------
  nnz.clear();
  int n_mesh_dofs = dof_handler[DH_MESH].numDofs();
  {
    std::vector<SetVector<int> > table;
    dof_handler[DH_MESH].getSparsityTable(table); // TODO: melhorar desempenho, função mt lenta

    nnz.resize(n_mesh_dofs, 0);

    //FEP_PRAGMA_OMP(parallel for)
    for (int i = 0; i < n_mesh_dofs; ++i)
      {nnz[i] = table[i].size();}  //cout << nnz[i] << " ";} //cout << endl;
  }

  //Mat Mat_Jac_m;
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_m);                                        CHKERRQ(ierr);
  ierr = MatSetType(Mat_Jac_m, MATSEQAIJ);                                               CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_m, PETSC_DECIDE, PETSC_DECIDE, n_mesh_dofs, n_mesh_dofs);   CHKERRQ(ierr);
  //ierr = MatSetFromOptions(Mat_Jac_m);                                                 CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_m,  0, nnz.data());                           CHKERRQ(ierr);
  //ierr = MatSetOption(Mat_Jac_m,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);                  CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);             CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_SYMMETRIC,PETSC_TRUE);                               CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_m);                                          CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_m, Vec_res_m, FormFunction_mesh, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_m, Mat_Jac_m, Mat_Jac_m, FormJacobian_mesh, this);         CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes_m,&linesearch);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);
  ierr = SNESGetKSP(snes_m,&ksp_m);                                                      CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_m,&pc_m);                                                          CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_m,Mat_Jac_m,Mat_Jac_m,SAME_NONZERO_PATTERN);                CHKERRQ(ierr);
  //ierr = KSPSetType(ksp_m,KSPGMRES);                                                     CHKERRQ(ierr); //gmres iterative method for mesh
  ierr = KSPSetType(ksp_m,KSPPREONLY);                                               CHKERRQ(ierr); //non-iteratuve method for mesh
  ierr = PCSetType(pc_m,PCLU);                                                           CHKERRQ(ierr); //LU preconditioner for mesh
  //ierr = PCFactorSetMatOrderingType(pc_m, MATORDERINGNATURAL);                     CHKERRQ(ierr);
  //ierr = KSPSetTolerances(ksp_m,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);  CHKERRQ(ierr);
  //ierr = SNESSetApplicationContext(snes_m,this);
  //ierr = SNESSetFromOptions(snes_m); CHKERRQ(ierr);  //prints Newton iterations information SNES Function norm
  if(false && !nonlinear_elasticity)  //for linear systems
  {
    ierr = SNESSetType(snes_m, SNESKSPONLY); CHKERRQ(ierr);
  }
 // cout << endl; ierr = SNESView(snes_m, PETSC_VIEWER_STDOUT_WORLD);
//~ #ifdef PETSC_HAVE_MUMPS
  //~ PCFactorSetMatSolverPackage(pc_m,MATSOLVERMUMPS);
//~ #endif

  ierr = SNESMonitorSet(snes_m, SNESMonitorDefault, 0, 0); CHKERRQ(ierr);
  //ierr = SNESMonitorSet(snes_m,Monitor,0,0);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes_m,1.e-11,1.e-11,1.e-11,10,PETSC_DEFAULT);
  //ierr = SNESSetFromOptions(snes_m); CHKERRQ(ierr);
  //ierr = SNESLineSearchSet(snes_m, SNESLineSearchNo, &user); CHKERRQ(ierr);
*/

  //  ------------------------------------------------------------------
  //  ----------------------- Fluid ------------------------------------
  //  ------------------------------------------------------------------

//  nnz.clear();
//  {
//    nnz.resize(n_unknowns_fs /*dof_handler[DH_UNKM].numDofs()+n_solids*LZ*/,0);
//    std::vector<SetVector<int> > tabla;
//    dof_handler[DH_UNKM].getSparsityTable(tabla); // TODO: melhorar desempenho, função mt lenta
//    //FEP_PRAGMA_OMP(parallel for)
//      for (int i = 0; i < n_unknowns_u + n_unknowns_p /*n_unknowns_fs - n_solids*LZ*/; ++i)
//        nnz[i] = tabla[i].size();
//  }
/*
    ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_fs);                                            CHKERRQ(ierr);
    ierr = MatSetSizes(Mat_Jac_fs, PETSC_DECIDE, PETSC_DECIDE, n_unknowns_fs, n_unknowns_fs);   CHKERRQ(ierr);
    ierr = MatSetFromOptions(Mat_Jac_fs);                                                       CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Mat_Jac_fs, 0, nnz.data());                                CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_fs,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);                 CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fs);                                              CHKERRQ(ierr);
    ierr = SNESSetFunction(snes_fs, Vec_res_fs, FormFunction_fs, this);                         CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes_fs, Mat_Jac_fs, Mat_Jac_fs, FormJacobian_fs, this);             CHKERRQ(ierr);
    ierr = SNESSetConvergenceTest(snes_fs,CheckSnesConvergence,this,PETSC_NULL);                CHKERRQ(ierr);
    ierr = SNESGetKSP(snes_fs,&ksp_fs);                                                         CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_fs,&pc_fs);                                                             CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_fs,Mat_Jac_fs,Mat_Jac_fs,SAME_NONZERO_PATTERN);                  CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes_fs); CHKERRQ(ierr);
    //cout << endl; ierr = SNESView(snes_fs, PETSC_VIEWER_STDOUT_WORLD);
*/

    if (is_sslv){
      allocPetscObjsSwimmer();
/*    //  ------------------------------------------------------------------
    //  ----------------------- Swimmer ----------------------------------
    //  ------------------------------------------------------------------

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_s);                             CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_res_s, PETSC_DECIDE, n_unknowns_sv);                 CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_res_s);                                        CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slip_rho);                          CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_slip_rho, PETSC_DECIDE, n_unknowns_sv);              CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_slip_rho);                                     CHKERRQ(ierr);
    ierr = VecSetOption(Vec_slip_rho, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_normal_aux);                        CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_normal_aux, PETSC_DECIDE, n_dofs_v_mesh);            CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_normal_aux);                                   CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_rho_aux);                           CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_rho_aux, PETSC_DECIDE, n_unknowns_sv);               CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_rho_aux);                                      CHKERRQ(ierr);

    nnz.clear();
    int n_pho_dofs = n_unknowns_sv;
    {
      std::vector<SetVector<int> > tabli;
      dof_handler[DH_SLIP].getSparsityTable(tabli); // TODO: melhorar desempenho, função mt lenta

      nnz.resize(n_pho_dofs, 0);

      //FEP_PRAGMA_OMP(parallel for)
      for (int i = 0; i < n_pho_dofs; ++i)
        {nnz[i] = tabli[i].size();}  //cout << nnz[i] << " ";} //cout << endl;
    }

    ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_s);                                         CHKERRQ(ierr);
    ierr = MatSetType(Mat_Jac_s, MATSEQAIJ);                                                CHKERRQ(ierr);
    ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, n_pho_dofs, n_pho_dofs);      CHKERRQ(ierr);
    //ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, 2*dim*n_solids, 2*dim*n_solids);     CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Mat_Jac_s,  0, nnz.data());                            CHKERRQ(ierr);
    //ierr = MatSetFromOptions(Mat_Jac_s);                                                CHKERRQ(ierr);
    //ierr = MatSeqAIJSetPreallocation(Mat_Jac_s, PETSC_DEFAULT, NULL);                   CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_s,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);              CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_s,MAT_SYMMETRIC,PETSC_TRUE);                                CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD, &snes_s);                                           CHKERRQ(ierr);
    ierr = SNESSetFunction(snes_s, Vec_res_s, FormFunction_sqrm, this);                     CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes_s, Mat_Jac_s, Mat_Jac_s, FormJacobian_sqrm, this);          CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes_s,&linesearch_s);
    ierr = SNESLineSearchSetType(linesearch_s,SNESLINESEARCHBASIC);
    ierr = SNESGetKSP(snes_s,&ksp_s);                                                       CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_s,&pc_s);                                                           CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_s,Mat_Jac_s,Mat_Jac_s,SAME_NONZERO_PATTERN);                 CHKERRQ(ierr);
    ierr = KSPSetType(ksp_s,KSPPREONLY);                                                    CHKERRQ(ierr);
    ierr = PCSetType(pc_s,PCLU);                                                            CHKERRQ(ierr);
    //ierr = SNESSetFromOptions(snes_s);                                                  CHKERRQ(ierr);
    ierr = SNESSetType(snes_s, SNESKSPONLY);                                                CHKERRQ(ierr);
    //cout << endl; ierr = SNESView(snes_s, PETSC_VIEWER_STDOUT_WORLD);
*/    }

    if (is_sfip){
      allocPetscObjsDissForce();
/*    //  ------------------------------------------------------------------
    //  ----------------------- Dissipative Force ------------------------
    //  ------------------------------------------------------------------

    //Vec Vec_fdis_0;
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_fdis_0);                            CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_fdis_0, PETSC_DECIDE, n_dofs_v_mesh);                CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_fdis_0);                                       CHKERRQ(ierr);
    ierr = VecSetOption(Vec_fdis_0, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);    CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_Fdis);                          CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_res_Fdis, PETSC_DECIDE, n_dofs_v_mesh);              CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_res_Fdis);                                     CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ftau_0);                            CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_ftau_0, PETSC_DECIDE, n_dofs_v_mesh);                CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_ftau_0);                                       CHKERRQ(ierr);
    ierr = VecSetOption(Vec_ftau_0, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);    CHKERRQ(ierr);

    nnz.clear();
    int n_fd_dofs = dof_handler[DH_MESH].numDofs();;
    {
      std::vector<SetVector<int> > tablf;
      dof_handler[DH_MESH].getSparsityTable(tablf); // TODO: melhorar desempenho, função mt lenta

      nnz.resize(n_fd_dofs, 0);

      //FEP_PRAGMA_OMP(parallel for)
      for (int i = 0; i < n_fd_dofs; ++i)
        {nnz[i] = tablf[i].size();}  //cout << nnz[i] << " ";} //cout << endl;
    }

    ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_fd);                                     CHKERRQ(ierr);
    ierr = MatSetType(Mat_Jac_fd, MATSEQAIJ);                                            CHKERRQ(ierr);
    ierr = MatSetSizes(Mat_Jac_fd, PETSC_DECIDE, PETSC_DECIDE, n_fd_dofs, n_fd_dofs);    CHKERRQ(ierr);
    //ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, 2*dim*n_solids, 2*dim*n_solids);     CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Mat_Jac_fd,  0, nnz.data());                        CHKERRQ(ierr);
    //ierr = MatSetFromOptions(Mat_Jac_s);                                                CHKERRQ(ierr);
    //ierr = MatSeqAIJSetPreallocation(Mat_Jac_s, PETSC_DEFAULT, NULL);                   CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_fd,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);          CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_fd,MAT_SYMMETRIC,PETSC_TRUE);                            CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fd);                                       CHKERRQ(ierr);
    ierr = SNESSetFunction(snes_fd, Vec_res_Fdis, FormFunction_fd, this);                CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes_fd, Mat_Jac_fd, Mat_Jac_fd, FormJacobian_fd, this);      CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes_fd,&linesearch_fd);
    ierr = SNESLineSearchSetType(linesearch_fd,SNESLINESEARCHBASIC);
    ierr = SNESGetKSP(snes_fd,&ksp_fd);                                                  CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_fd,&pc_fd);                                                      CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_fd,Mat_Jac_fd,Mat_Jac_fd,SAME_NONZERO_PATTERN);           CHKERRQ(ierr);
    ierr = KSPSetType(ksp_fd,KSPGMRES);                                                CHKERRQ(ierr); //KSPPREONLY
    ierr = PCSetType(pc_fd,PCLU);                                                        CHKERRQ(ierr);
    //ierr = SNESSetFromOptions(snes_fd);                                                  CHKERRQ(ierr);
    //ierr = SNESSetType(snes_fd, SNESKSPONLY);                                            CHKERRQ(ierr);
    //ierr = SNESView(snes_fd, PETSC_VIEWER_STDOUT_WORLD);
    ierr = SNESMonitorSet(snes_fd, SNESMonitorDefault, 0, 0);                            CHKERRQ(ierr);
    //ierr = SNESMonitorSet(snes_m,Monitor,0,0);CHKERRQ(ierr);
    ierr = SNESSetTolerances(snes_fd,1.e-11,1.e-11,1.e-11,10,PETSC_DEFAULT);             CHKERRQ(ierr);
*/    }

  printf(" done.\n");
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::allocPetscObjsVecInt()
{
  //  ------------------------------------------------------------------
  //  ---------- Interpolated Vectors in Mesh Adaptation ---------------
  //  ------------------------------------------------------------------
  PetscErrorCode      ierr;

  //Vec Vec_ups_0;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ups_0);                               CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_ups_0, PETSC_DECIDE, n_unknowns_fs);                   CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_ups_0);                                          CHKERRQ(ierr);
  ierr = VecSetOption(Vec_ups_0, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);       CHKERRQ(ierr);

  //Vec Vec_ups_1;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ups_1);                               CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_ups_1, PETSC_DECIDE, n_unknowns_fs);                   CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_ups_1);                                          CHKERRQ(ierr);
  ierr = VecSetOption(Vec_ups_1, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);       CHKERRQ(ierr);

  //Vec Vec_x_0;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_0);                  CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_x_0, PETSC_DECIDE, n_dofs_v_mesh);      CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_x_0);                             CHKERRQ(ierr);

  //Vec Vec_x_1;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_1);                  CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_x_1, PETSC_DECIDE, n_dofs_v_mesh);      CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_x_1);                             CHKERRQ(ierr);

  //Vec Vec_ups_m1;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ups_m1);                              CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_ups_m1, PETSC_DECIDE, n_unknowns_fs);                  CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_ups_m1);                                         CHKERRQ(ierr);
  ierr = VecSetOption(Vec_ups_m1, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);      CHKERRQ(ierr);

  //Vec Vec_x_aux;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_aux);              CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_x_aux, PETSC_DECIDE, n_dofs_v_mesh);  CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_x_aux);                         CHKERRQ(ierr);

  if (is_bdf3)
  {
    //Vec Vec_ups_m2;
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ups_m2);                            CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_ups_m2, PETSC_DECIDE, n_unknowns_fs);                CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_ups_m2);                                       CHKERRQ(ierr);
    ierr = VecSetOption(Vec_ups_m2, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);    CHKERRQ(ierr);
  }

  if (is_slipv)
  {
    //Vec Vec_slipvel_0;
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slipv_0);                           CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_slipv_0, PETSC_DECIDE, n_dofs_v_mesh);               CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_slipv_0);                                      CHKERRQ(ierr);
    ierr = VecSetOption(Vec_slipv_0, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);   CHKERRQ(ierr);

    //Vec Vec_slipvel_1;
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slipv_1);                           CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_slipv_1, PETSC_DECIDE, n_dofs_v_mesh);               CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_slipv_1);                                      CHKERRQ(ierr);
    ierr = VecSetOption(Vec_slipv_1, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);   CHKERRQ(ierr);

    //Vec Vec_slipvel_m1;
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slipv_m1);                          CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_slipv_m1, PETSC_DECIDE, n_dofs_v_mesh);              CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_slipv_m1);                                     CHKERRQ(ierr);
    ierr = VecSetOption(Vec_slipv_m1, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);

    if (is_bdf3)
    {
      //Vec Vec_slipvel_m2;
      ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slipv_m2);                        CHKERRQ(ierr);
      ierr = VecSetSizes(Vec_slipv_m2, PETSC_DECIDE, n_dofs_v_mesh);            CHKERRQ(ierr);
      ierr = VecSetFromOptions(Vec_slipv_m2);                                   CHKERRQ(ierr);
      ierr = VecSetOption(Vec_slipv_m2, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::allocPetscObjsVecNoInt()
{
  //  ------------------------------------------------------------------
  //  --------No Interpolated Vectors in Mesh Adaptation ---------------
  //  ------------------------------------------------------------------
  PetscErrorCode      ierr;

  //Vec Vec_res;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_fs);               CHKERRQ(ierr);  //creating Vec_res empty PETSc vector
  ierr = VecSetSizes(Vec_res_fs, PETSC_DECIDE, n_unknowns_fs);   CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_res_fs);                          CHKERRQ(ierr);

  //Vec Vec_res_m;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_m);                CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_res_m, PETSC_DECIDE, n_dofs_v_mesh);    CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_res_m);                           CHKERRQ(ierr);

  //Vec Vec_v_mid
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_v_mid);                CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_v_mid, PETSC_DECIDE, n_dofs_v_mesh);    CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_v_mid);                           CHKERRQ(ierr);

  //Vec Vec_v_1
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_v_1);                  CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_v_1, PETSC_DECIDE, n_dofs_v_mesh);      CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_v_1);                             CHKERRQ(ierr);

  //Vec Vec_normal;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_normal);               CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_normal, PETSC_DECIDE, n_dofs_v_mesh);   CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_normal);                          CHKERRQ(ierr);

  //Vec Vec_tangent;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_tangent);              CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_tangent, PETSC_DECIDE, n_dofs_v_mesh);  CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_tangent);                         CHKERRQ(ierr);

  //if (dim == 3){
  //  //Vec Vec_tangent;
  //  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_binormal);               CHKERRQ(ierr);
  //  ierr = VecSetSizes(Vec_binormal, PETSC_DECIDE, n_dofs_v_mesh);   CHKERRQ(ierr);
  //  ierr = VecSetFromOptions(Vec_binormal);                          CHKERRQ(ierr);
  //}

  if (is_bdf2 || is_bdf3)
  {
    //Vec Vec_x_cur; current
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_cur);              CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_x_cur, PETSC_DECIDE, n_dofs_v_mesh);  CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_x_cur);                         CHKERRQ(ierr);
  }

  if (is_sfip){
    //Vec Vec_metav_0; current
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_metav_0);              CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_metav_0, PETSC_DECIDE, n_dofs_v_mesh);  CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_metav_0);                         CHKERRQ(ierr);
  }

  if (time_adapt)
  {
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ups_time_aux);                          CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_ups_time_aux, PETSC_DECIDE, n_unknowns_fs);              CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_ups_time_aux);                                     CHKERRQ(ierr);
    ierr = VecSetOption(Vec_ups_time_aux, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_time_aux);                            CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_x_time_aux, PETSC_DECIDE, n_dofs_v_mesh);                CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_x_time_aux);                                       CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::allocPetscObjsMesh()
{
  //  ------------------------------------------------------------------
  //  ----------------------- Mesh -------------------------------------
  //  ------------------------------------------------------------------
  PetscErrorCode      ierr;

  std::vector<int> nnz;
  nnz.clear();
  int n_mesh_dofs = dof_handler[DH_MESH].numDofs();
  {
    std::vector<SetVector<int> > table;
    dof_handler[DH_MESH].getSparsityTable(table); // TODO: melhorar desempenho, função mt lenta

    nnz.resize(n_mesh_dofs, 0);

    //FEP_PRAGMA_OMP(parallel for)
    for (int i = 0; i < n_mesh_dofs; ++i)
      {nnz[i] = table[i].size();}  //cout << nnz[i] << " ";} //cout << endl;
  }

  //Mat Mat_Jac_m;
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_m);                                        CHKERRQ(ierr);
  ierr = MatSetType(Mat_Jac_m, MATSEQAIJ);                                               CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_m, PETSC_DECIDE, PETSC_DECIDE, n_mesh_dofs, n_mesh_dofs);   CHKERRQ(ierr);
  //ierr = MatSetFromOptions(Mat_Jac_m);                                                 CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_m,  0, nnz.data());                           CHKERRQ(ierr);
  //ierr = MatSetOption(Mat_Jac_m,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);                  CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);             CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_SYMMETRIC,PETSC_TRUE);                               CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_m);                                          CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_m, Vec_res_m, FormFunction_mesh, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_m, Mat_Jac_m, Mat_Jac_m, FormJacobian_mesh, this);         CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes_m,&linesearch);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);
  ierr = SNESGetKSP(snes_m,&ksp_m);                                                      CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_m,&pc_m);                                                          CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_m,Mat_Jac_m,Mat_Jac_m,SAME_NONZERO_PATTERN);                CHKERRQ(ierr);
  ierr = KSPSetType(ksp_m,KSPGMRES);                                                     CHKERRQ(ierr); //gmres iterative method for mesh
  //ierr = KSPSetType(ksp_m,KSPPREONLY);                                               CHKERRQ(ierr); //non-iteratuve method for mesh
  ierr = PCSetType(pc_m,PCLU);                                                           CHKERRQ(ierr); //LU preconditioner for mesh
  //ierr = PCFactorSetMatOrderingType(pc_m, MATORDERINGNATURAL);                     CHKERRQ(ierr);
  //ierr = KSPSetTolerances(ksp_m,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);  CHKERRQ(ierr);
  //ierr = SNESSetApplicationContext(snes_m,this);
  //ierr = SNESSetFromOptions(snes_m); CHKERRQ(ierr);  //prints Newton iterations information SNES Function norm
  if(false && !nonlinear_elasticity)  //for linear systems
  {
    ierr = SNESSetType(snes_m, SNESKSPONLY); CHKERRQ(ierr);
  }
 // cout << endl; ierr = SNESView(snes_m, PETSC_VIEWER_STDOUT_WORLD);
//~ #ifdef PETSC_HAVE_MUMPS
  //~ PCFactorSetMatSolverPackage(pc_m,MATSOLVERMUMPS);
//~ #endif

  ierr = SNESMonitorSet(snes_m, SNESMonitorDefault, 0, 0); CHKERRQ(ierr);
  //ierr = SNESMonitorSet(snes_m,Monitor,0,0);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes_m,1.e-11,1.e-11,1.e-11,10,PETSC_DEFAULT);
  //ierr = SNESSetFromOptions(snes_m); CHKERRQ(ierr);
  //ierr = SNESLineSearchSet(snes_m, SNESLineSearchNo, &user); CHKERRQ(ierr);
  ierr = KSPMonitorSet(ksp_m, KSPMonitorDefault, 0, 0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::allocPetscObjsFluid()
{
  //  ------------------------------------------------------------------
  //  ----------------------- Fluid ------------------------------------
  //  ------------------------------------------------------------------
  PetscErrorCode      ierr;

  std::vector<int> nnz;
  nnz.clear();
  {
    nnz.resize(n_unknowns_fs /*dof_handler[DH_UNKM].numDofs()+n_solids*LZ*/,0);
    std::vector<SetVector<int> > tabla;
    dof_handler[DH_UNKM].getSparsityTable(tabla); // TODO: melhorar desempenho, função mt lenta
    //FEP_PRAGMA_OMP(parallel for)
      for (int i = 0; i < n_unknowns_u + n_unknowns_p /*n_unknowns_fs - n_solids*LZ*/; ++i)
        nnz[i] = tabla[i].size();
  }

  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_fs);                                            CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_fs, PETSC_DECIDE, PETSC_DECIDE, n_unknowns_fs, n_unknowns_fs);   CHKERRQ(ierr);
  ierr = MatSetFromOptions(Mat_Jac_fs);                                                       CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_fs, 0, nnz.data());                                CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_fs,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);                 CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fs);                                              CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_fs, Vec_res_fs, FormFunction_fs, this);                         CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_fs, Mat_Jac_fs, Mat_Jac_fs, FormJacobian_fs, this);             CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(snes_fs,CheckSnesConvergence,this,PETSC_NULL);                CHKERRQ(ierr);
  ierr = SNESGetKSP(snes_fs,&ksp_fs);                                                         CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_fs,&pc_fs);                                                             CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_fs,Mat_Jac_fs,Mat_Jac_fs,SAME_NONZERO_PATTERN);                  CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_fs); CHKERRQ(ierr);
  //cout << endl; ierr = SNESView(snes_fs, PETSC_VIEWER_STDOUT_WORLD);

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::allocPetscObjsSwimmer()
{
  //  ------------------------------------------------------------------
  //  ----------------------- Swimmer ----------------------------------
  //  ------------------------------------------------------------------
  PetscErrorCode      ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_s);                             CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_res_s, PETSC_DECIDE, n_unknowns_sv);                 CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_res_s);                                        CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slip_rho);                          CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_slip_rho, PETSC_DECIDE, n_unknowns_sv);              CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_slip_rho);                                     CHKERRQ(ierr);
  ierr = VecSetOption(Vec_slip_rho, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_normal_aux);                        CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_normal_aux, PETSC_DECIDE, n_dofs_v_mesh);            CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_normal_aux);                                   CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_rho_aux);                           CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_rho_aux, PETSC_DECIDE, n_unknowns_sv);               CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_rho_aux);                                      CHKERRQ(ierr);

  std::vector<int> nnz;
  nnz.clear();
  int n_pho_dofs = n_unknowns_sv;
  {
    std::vector<SetVector<int> > tabli;
    dof_handler[DH_SLIP].getSparsityTable(tabli); // TODO: melhorar desempenho, função mt lenta

    nnz.resize(n_pho_dofs, 0);

    //FEP_PRAGMA_OMP(parallel for)
    for (int i = 0; i < n_pho_dofs; ++i)
      {nnz[i] = tabli[i].size();}  //cout << nnz[i] << " ";} //cout << endl;
  }

  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_s);                                         CHKERRQ(ierr);
  ierr = MatSetType(Mat_Jac_s, MATSEQAIJ);                                                CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, n_pho_dofs, n_pho_dofs);      CHKERRQ(ierr);
  //ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, 2*dim*n_solids, 2*dim*n_solids);     CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_s,  0, nnz.data());                            CHKERRQ(ierr);
  //ierr = MatSetFromOptions(Mat_Jac_s);                                                CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(Mat_Jac_s, PETSC_DEFAULT, NULL);                   CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_s,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);              CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_s,MAT_SYMMETRIC,PETSC_TRUE);                                CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_s);                                           CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_s, Vec_res_s, FormFunction_sqrm, this);                     CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_s, Mat_Jac_s, Mat_Jac_s, FormJacobian_sqrm, this);          CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes_s,&linesearch_s);
  ierr = SNESLineSearchSetType(linesearch_s,SNESLINESEARCHBASIC);
  ierr = SNESGetKSP(snes_s,&ksp_s);                                                       CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_s,&pc_s);                                                           CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_s,Mat_Jac_s,Mat_Jac_s,SAME_NONZERO_PATTERN);                 CHKERRQ(ierr);
  ierr = KSPSetType(ksp_s,KSPPREONLY);                                                    CHKERRQ(ierr);
  ierr = PCSetType(pc_s,PCLU);                                                            CHKERRQ(ierr);
  //ierr = SNESSetFromOptions(snes_s);                                                  CHKERRQ(ierr);
  ierr = SNESSetType(snes_s, SNESKSPONLY);                                                CHKERRQ(ierr);
  //cout << endl; ierr = SNESView(snes_s, PETSC_VIEWER_STDOUT_WORLD);

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::allocPetscObjsDissForce()
{
  //  ------------------------------------------------------------------
  //  ----------------------- Dissipative Force ------------------------
  //  ------------------------------------------------------------------
  PetscErrorCode      ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_Fdis);                          CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_res_Fdis, PETSC_DECIDE, n_dofs_v_mesh);              CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_res_Fdis);                                     CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_fdis_0);                            CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_fdis_0, PETSC_DECIDE, n_dofs_v_mesh);                CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_fdis_0);                                       CHKERRQ(ierr);
  ierr = VecSetOption(Vec_fdis_0, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);    CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ftau_0);                            CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_ftau_0, PETSC_DECIDE, n_dofs_v_mesh);                CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_ftau_0);                                       CHKERRQ(ierr);
  ierr = VecSetOption(Vec_ftau_0, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);    CHKERRQ(ierr);

  std::vector<int> nnz;
  nnz.clear();
  int n_fd_dofs = dof_handler[DH_MESH].numDofs();;
  {
    std::vector<SetVector<int> > tablf;
    dof_handler[DH_MESH].getSparsityTable(tablf); // TODO: melhorar desempenho, função mt lenta

    nnz.resize(n_fd_dofs, 0);

    //FEP_PRAGMA_OMP(parallel for)
    for (int i = 0; i < n_fd_dofs; ++i)
      {nnz[i] = tablf[i].size();}  //cout << nnz[i] << " ";} //cout << endl;
  }

  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_fd);                                     CHKERRQ(ierr);
  ierr = MatSetType(Mat_Jac_fd, MATSEQAIJ);                                            CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_fd, PETSC_DECIDE, PETSC_DECIDE, n_fd_dofs, n_fd_dofs);    CHKERRQ(ierr);
  //ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, 2*dim*n_solids, 2*dim*n_solids);     CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_fd,  0, nnz.data());                        CHKERRQ(ierr);
  //ierr = MatSetFromOptions(Mat_Jac_s);                                                CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(Mat_Jac_s, PETSC_DEFAULT, NULL);                   CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_fd,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);          CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_fd,MAT_SYMMETRIC,PETSC_TRUE);                            CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fd);                                       CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_fd, Vec_res_Fdis, FormFunction_fd, this);                CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_fd, Mat_Jac_fd, Mat_Jac_fd, FormJacobian_fd, this);      CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes_fd,&linesearch_fd);
  ierr = SNESLineSearchSetType(linesearch_fd,SNESLINESEARCHBASIC);
  ierr = SNESGetKSP(snes_fd,&ksp_fd);                                                  CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_fd,&pc_fd);                                                      CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_fd,Mat_Jac_fd,Mat_Jac_fd,SAME_NONZERO_PATTERN);           CHKERRQ(ierr);
  ierr = KSPSetType(ksp_fd,KSPGMRES);                                                CHKERRQ(ierr); //KSPPREONLY
  ierr = PCSetType(pc_fd,PCLU);                                                        CHKERRQ(ierr);
  //ierr = SNESSetFromOptions(snes_fd);                                                  CHKERRQ(ierr);
  //ierr = SNESSetType(snes_fd, SNESKSPONLY);                                            CHKERRQ(ierr);
  //ierr = SNESView(snes_fd, PETSC_VIEWER_STDOUT_WORLD);
  ierr = SNESMonitorSet(snes_fd, SNESMonitorDefault, 0, 0);                            CHKERRQ(ierr);
  //ierr = SNESMonitorSet(snes_m,Monitor,0,0);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes_fd,1.e-14,1.e-14,1.e-14,100,PETSC_DEFAULT);             CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp_fd,1.e-16,1.e-16,1.e-16,100);             CHKERRQ(ierr);
  ierr = KSPMonitorSet(ksp_fd, KSPMonitorDefault, 0, 0); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

void AppCtx::matrixColoring()
{
  printf("Matrix coloring... ");  //clock_t t; t = clock();

  VectorXi                     mapU_c(n_dofs_u_per_cell);
  VectorXi                     mapZ_c(nodes_per_cell*LZ);
  VectorXi                     mapP_c(n_dofs_p_per_cell);

  //Z Dofs re-organization
  int tag;
  int nod_id, nod_is, nod_vs, nodsum;

  MatrixXd Afsloc = MatrixXd::Zero(n_dofs_u_per_cell, n_dofs_u_per_cell);
  MatrixXd Dfsloc = MatrixXd::Zero(n_dofs_p_per_cell, n_dofs_u_per_cell);
  MatrixXd Gfsloc = MatrixXd::Zero(n_dofs_u_per_cell, n_dofs_p_per_cell);
  MatrixXd Efsloc = MatrixXd::Zero(n_dofs_p_per_cell, n_dofs_p_per_cell);

  MatrixXd Z1fsloc = MatrixXd::Zero(n_dofs_u_per_cell,nodes_per_cell*LZ);
  MatrixXd Z2fsloc = MatrixXd::Zero(nodes_per_cell*LZ,n_dofs_u_per_cell);
  MatrixXd Z3fsloc = MatrixXd::Zero(nodes_per_cell*LZ,nodes_per_cell*LZ);
  MatrixXd Z4fsloc = MatrixXd::Zero(nodes_per_cell*LZ,n_dofs_p_per_cell);
  MatrixXd Z5fsloc = MatrixXd::Zero(n_dofs_p_per_cell,nodes_per_cell*LZ);

  std::vector<bool>   SV(n_solids,false);  //solid visited history
  bool                SFI = false;         //solid-fluid interception

  cell_iterator cell = mesh->cellBegin();
  cell_iterator cell_end = mesh->cellEnd();

  for (; cell != cell_end; ++cell){
    // mapeamento do local para o global:
    dof_handler[DH_UNKM].getVariable(VAR_U).getCellDofs(mapU_c.data(), &*cell);  // U global ids for the current cell
    dof_handler[DH_UNKM].getVariable(VAR_P).getCellDofs(mapP_c.data(), &*cell);  // P global ids for the current cell
    // mapping correction for velocity unknowns case FS
    if (is_sfip){
      mapZ_c = -VectorXi::Ones(nodes_per_cell*LZ);
      SFI = false;
      for (int j = 0; j < nodes_per_cell; ++j){
        tag = mesh->getNodePtr(cell->getNodeId(j))->getTag();
        nod_id = is_in_id(tag,flusoli_tags);
        nod_is = is_in_id(tag,solidonly_tags);
        nod_vs = is_in_id(tag,slipvel_tags);
        nodsum = nod_id+nod_is+nod_vs;
        if (nodsum){
          for (int l = 0; l < LZ; l++){
            mapZ_c(j*LZ + l) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + l;
          }
          for (int l = 0; l < dim; l++){
            mapU_c(j*dim + l) = -1;
          }
          SFI = true;
        }
      }
    }
    //cout << endl << mapU_c.transpose() << "   "<< mapZ_c.transpose() << endl;
    MatSetValues(Mat_Jac_fs, mapU_c.size(), mapU_c.data(), mapU_c.size(), mapU_c.data(), Afsloc.data(), ADD_VALUES);
    MatSetValues(Mat_Jac_fs, mapU_c.size(), mapU_c.data(), mapP_c.size(), mapP_c.data(), Gfsloc.data(), ADD_VALUES);
    MatSetValues(Mat_Jac_fs, mapP_c.size(), mapP_c.data(), mapU_c.size(), mapU_c.data(), Dfsloc.data(), ADD_VALUES);
    MatSetValues(Mat_Jac_fs, mapP_c.size(), mapP_c.data(), mapP_c.size(), mapP_c.data(), Efsloc.data(), ADD_VALUES);
    if (SFI){
      MatSetValues(Mat_Jac_fs, mapU_c.size(), mapU_c.data(), mapZ_c.size(), mapZ_c.data(), Z1fsloc.data(), ADD_VALUES);
      MatSetValues(Mat_Jac_fs, mapZ_c.size(), mapZ_c.data(), mapU_c.size(), mapU_c.data(), Z2fsloc.data(), ADD_VALUES);
      MatSetValues(Mat_Jac_fs, mapZ_c.size(), mapZ_c.data(), mapZ_c.size(), mapZ_c.data(), Z3fsloc.data(), ADD_VALUES);
      MatSetValues(Mat_Jac_fs, mapZ_c.size(), mapZ_c.data(), mapP_c.size(), mapP_c.data(), Z4fsloc.data(), ADD_VALUES);
      MatSetValues(Mat_Jac_fs, mapP_c.size(), mapP_c.data(), mapZ_c.size(), mapZ_c.data(), Z5fsloc.data(), ADD_VALUES);
    }
  }//end for cell

  Assembly(Mat_Jac_fs); //View(Mat_Jac_fs,"MatColored.m","JJ");

  printf(" done. \n");  //t = clock() - t; printf ("It took me %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
}

void AppCtx::printMatlabLoader()
{
  FILE *fp = fopen("loadmat.m", "w");
  //fprintf(fp, "clear;\n"                   );
  fprintf(fp, "jacob;\n"                   );
  fprintf(fp, "clear zzz;\n"               );
  fprintf(fp, "B=Jac;\n"                   );
  fprintf(fp, "B(B!=0)=1;\n"               );
  fprintf(fp, "nU = %d;\n",dof_handler[DH_UNKM].getVariable(VAR_U).numPositiveDofs() );
  fprintf(fp, "nP = %d;\n",dof_handler[DH_UNKM].getVariable(VAR_P).numPositiveDofs() );
  fprintf(fp, "nT = nU + nP;\n"            );
  fprintf(fp, "K=Jac(1:nU,1:nU);\n"        );
  fprintf(fp, "G=Jac(1:nU,nU+1:nT);\n"     );
  fprintf(fp, "D=Jac(nU+1:nT,1:nU);\n"     );
  fprintf(fp, "E=Jac(nU+1:nT,nU+1:nT);\n"  );
  fprintf(fp, "\n"                        );
  fprintf(fp, "rhs;\n"                     );
  fprintf(fp, "f=res(1:nU);\n"             );
  fprintf(fp, "g=res(nU+1:nT);\n"          );
  fclose(fp);
}

// must be called after loadDofs
void AppCtx::evaluateQuadraturePts()
{
  // avaliando phi_c e as matrizes de transf nos pontos de quadratura.
  phi_c.resize(n_qpts_cell);
  psi_c.resize(n_qpts_cell);
  qsi_c.resize(n_qpts_cell);

  phi_f.resize(n_qpts_facet);
  psi_f.resize(n_qpts_facet);
  qsi_f.resize(n_qpts_facet);

  phi_r.resize(n_qpts_corner);
  psi_r.resize(n_qpts_corner);
  qsi_r.resize(n_qpts_corner);

  dLphi_c.resize(n_qpts_cell);
  dLpsi_c.resize(n_qpts_cell);
  dLqsi_c.resize(n_qpts_cell);

  dLphi_f.resize(n_qpts_facet);
  dLpsi_f.resize(n_qpts_facet);
  dLqsi_f.resize(n_qpts_facet);

  dLphi_r.resize(n_qpts_corner);
  dLpsi_r.resize(n_qpts_corner);
  dLqsi_r.resize(n_qpts_corner);

  bble.resize(n_qpts_cell);
  dLbble.resize(n_qpts_cell);

  for (int qp = 0; qp < n_qpts_cell; ++qp)
  {
    phi_c[qp].resize(n_dofs_u_per_cell/dim);
    psi_c[qp].resize(n_dofs_p_per_cell);
    qsi_c[qp].resize(nodes_per_cell);

    dLphi_c[qp].resize(n_dofs_u_per_cell/dim, dim);
    dLpsi_c[qp].resize(n_dofs_p_per_cell, dim);
    dLqsi_c[qp].resize(nodes_per_cell, dim);

    dLbble[qp].resize(dim);
    bble[qp] = shape_bble->eval(quadr_cell->point(qp), 0);

    for (int n = 0; n < n_dofs_u_per_cell/dim; ++n)
    {
      phi_c[qp][n] = shape_phi_c->eval(quadr_cell->point(qp), n); //std::cout << phi_c[qp][n] << "\n";
      for (int d = 0; d < dim; ++d)
      {
        /* dLphi_c nao depende de qp no caso de funcoes lineares */
        dLphi_c[qp](n, d) = shape_phi_c->gradL(quadr_cell->point(qp), n, d);
      }
    }

    for (int n = 0; n < n_dofs_p_per_cell; ++n)
    {
      psi_c[qp][n] = shape_psi_c->eval(quadr_cell->point(qp), n);
      for (int d = 0; d < dim; ++d)
      {
        /* dLpsi_c nao depende de qp no caso de funcoes lineares */
        dLpsi_c[qp](n, d) = shape_psi_c->gradL(quadr_cell->point(qp), n, d);
      }
    }

    for (int n = 0; n < nodes_per_cell; ++n)
    {
      qsi_c[qp][n] = shape_qsi_c->eval(quadr_cell->point(qp), n);
      for (int d = 0; d < dim; ++d)
      {
        /* dLqsi_c nao depende de qp no caso de funcoes lineares */
        dLqsi_c[qp](n, d) = shape_qsi_c->gradL(quadr_cell->point(qp), n, d);
      }
    }

    for (int d = 0; d < dim; ++d)
    {
      dLbble[qp](d) = shape_bble->gradL(quadr_cell->point(qp), 0, d);
    }

  } // end qp

  // pts de quadratura no contorno
  for (int qp = 0; qp < n_qpts_facet; ++qp)
  {
    phi_f[qp].resize(n_dofs_u_per_facet/dim);
    psi_f[qp].resize(n_dofs_p_per_facet);
    qsi_f[qp].resize(n_dofs_v_per_facet/dim);

    dLphi_f[qp].resize(n_dofs_u_per_facet/dim, dim-1);
    dLpsi_f[qp].resize(n_dofs_p_per_facet, dim-1);
    dLqsi_f[qp].resize(n_dofs_v_per_facet/dim, dim-1);

    for (int n = 0; n < n_dofs_u_per_facet/dim; ++n)
    {
      phi_f[qp][n] = shape_phi_f->eval(quadr_facet->point(qp), n);
      for (int d = 0; d < dim-1; ++d)
      {
        /* dLphi_f nao depende de qp no caso de funcoes lineares */
        dLphi_f[qp](n, d) = shape_phi_f->gradL(quadr_facet->point(qp), n, d);
      }
    }

    for (int n = 0; n < n_dofs_p_per_facet; ++n)
    {
      psi_f[qp][n] = shape_psi_f->eval(quadr_facet->point(qp), n);
      for (int d = 0; d < dim-1; ++d)
      {
        /* dLpsi_f nao depende de qp no caso de funcoes lineares */
        dLpsi_f[qp](n, d) = shape_psi_f->gradL(quadr_facet->point(qp), n, d);
      }
    }

    for (int n = 0; n < nodes_per_facet; ++n)
    {
      qsi_f[qp][n] = shape_qsi_f->eval(quadr_facet->point(qp), n);
      for (int d = 0; d < dim-1; ++d)
      {
        /* dLqsi_f nao depende de qp no caso de funcoes lineares */
        dLqsi_f[qp](n, d) = shape_qsi_f->gradL(quadr_facet->point(qp), n, d);
      }
    }

  } // end qp

  // pts de quadratura no contorno do contorno
  for (int qp = 0; qp < n_qpts_corner; ++qp)
  {
    phi_r[qp].resize(n_dofs_u_per_corner/dim);
    psi_r[qp].resize(n_dofs_p_per_corner);
    qsi_r[qp].resize(nodes_per_corner);

    dLphi_r[qp].resize(n_dofs_u_per_corner/dim, 1 /* = dim-2*/);
    dLpsi_r[qp].resize(n_dofs_p_per_corner, 1 /* = dim-2*/);
    dLqsi_r[qp].resize(nodes_per_corner, 1 /* = dim-2*/);

    for (int n = 0; n < n_dofs_u_per_corner/dim; ++n)
    {
      phi_r[qp][n] = shape_phi_r->eval(quadr_corner->point(qp), n);
      for (int d = 0; d < dim-2; ++d)
      {
        /* dLphi_r nao depende de qp no caso de funcoes lineares */
        dLphi_r[qp](n, d) = shape_phi_r->gradL(quadr_corner->point(qp), n, d);
      }
    }

    for (int n = 0; n < n_dofs_p_per_corner; ++n)
    {
      psi_r[qp][n] = shape_psi_r->eval(quadr_corner->point(qp), n);
      for (int d = 0; d < dim-2; ++d)
      {
        /* dLpsi_r nao depende de qp no caso de funcoes lineares */
        dLpsi_r[qp](n, d) = shape_psi_r->gradL(quadr_corner->point(qp), n, d);
      }
    }

    for (int n = 0; n < nodes_per_corner; ++n)
    {
      qsi_r[qp][n] = shape_qsi_r->eval(quadr_corner->point(qp), n);
      for (int d = 0; d < dim-2; ++d)
      {
        /* dLqsi_r nao depende de qp no caso de funcoes lineares */
        dLqsi_r[qp](n, d) = shape_qsi_r->gradL(quadr_corner->point(qp), n, d);
      }
    }
  } // end qp

  //     Quadrature
  //     to compute the error
  //
  // velocity
  phi_err.resize(n_qpts_err);         // shape function evaluated at quadrature points
  dLphi_err.resize(n_qpts_err);       // matriz de gradiente no elemento unitário
  // pressure
  psi_err.resize(n_qpts_err);         // shape function evaluated at quadrature points
  dLpsi_err.resize(n_qpts_err);       // matriz de gradiente no elemento unitário
  // mesh
  qsi_err.resize(n_qpts_err);         // shape function evaluated at quadrature points
  dLqsi_err.resize(n_qpts_err);       // matriz de gradiente no elemento unitário

  for (int qp = 0; qp < n_qpts_err; ++qp)
  {
    phi_err[qp].resize(n_dofs_u_per_cell/dim);
    psi_err[qp].resize(n_dofs_p_per_cell);
    qsi_err[qp].resize(nodes_per_cell);

    dLphi_err[qp].resize(n_dofs_u_per_cell/dim, dim);
    dLpsi_err[qp].resize(n_dofs_p_per_cell, dim);
    dLqsi_err[qp].resize(nodes_per_cell, dim);

    for (int n = 0; n < n_dofs_u_per_cell/dim; ++n)
    {
      phi_err[qp][n] = shape_phi_c->eval(quadr_err->point(qp), n);
      for (int d = 0; d < dim; ++d)
      {
        /* dLphi_err nao depende de qp no caso de funcoes lineares */
        dLphi_err[qp](n, d) = shape_phi_c->gradL(quadr_err->point(qp), n, d);
      }
    }

    for (int n = 0; n < n_dofs_p_per_cell; ++n)
    {
      psi_err[qp][n] = shape_psi_c->eval(quadr_err->point(qp), n);
      for (int d = 0; d < dim; ++d)
      {
        /* dLpsi_err nao depende de qp no caso de funcoes lineares */
        dLpsi_err[qp](n, d) = shape_psi_c->gradL(quadr_err->point(qp), n, d);
      }
    }

    for (int n = 0; n < nodes_per_cell; ++n)
    {
      qsi_err[qp][n] = shape_qsi_c->eval(quadr_err->point(qp), n);
      for (int d = 0; d < dim; ++d)
      {
        /* dLqsi_err nao depende de qp no caso de funcoes lineares */
        dLqsi_err[qp](n, d) = shape_qsi_c->gradL(quadr_err->point(qp), n, d);
      }
    }

  } // end qp

  Vector Xc = Vector::Zero(dim);
  bool is_simplex = ctype2cfamily(ECellType(mesh_cell_type)) == SIMPLEX;
  if (is_simplex)
    for (int i = 0; i < dim; ++i)
      Xc(i) = 1./(dim+1);
  qsi_c_at_center.resize(nodes_per_cell);
  for (int i = 0; i < nodes_per_cell; ++i)
    qsi_c_at_center(i) = shape_qsi_c->eval(Xc.data(), i);

  // facets func derivatives on your nodes
  if (dim==2)
  {
    std::vector<double> parametric_pts;
    parametric_pts = genLineParametricPts(  ctypeDegree(ECellType(mesh_cell_type))  );

    dLphi_nf.resize(parametric_pts.size());

    for (int k = 0; k < (int)parametric_pts.size(); ++k)
    {
      dLphi_nf[k].resize(n_dofs_u_per_facet/dim, dim-1);
      for (int n = 0; n < n_dofs_u_per_facet/dim; ++n)
        for (int d = 0; d < dim-1; ++d)
          dLphi_nf[k](n, d) = shape_phi_f->gradL(&parametric_pts[k], n, d);
    }
  }
  else
  if(dim==3)
  {
    std::vector<Eigen::Vector2d> parametric_pts;
    parametric_pts = genTriParametricPts(  ctypeDegree(ECellType(mesh_cell_type))  );

    dLphi_nf.resize(parametric_pts.size());

    for (int k = 0; k < (int)parametric_pts.size(); ++k)
    {
      dLphi_nf[k].resize(n_dofs_u_per_facet/dim, dim-1);
      for (int n = 0; n < n_dofs_u_per_facet/dim; ++n)
        for (int d = 0; d < dim-1; ++d)
          dLphi_nf[k](n, d) = shape_phi_f->gradL(parametric_pts[k].data(), n, d);
    }
  }

}

void AppCtx::onUpdateMesh()
{
  allocPetscObjs();
  matrixColoring();
}

PetscErrorCode AppCtx::checkSnesConvergence(SNES snes, PetscInt it,PetscReal xnorm, PetscReal pnorm, PetscReal fnorm, SNESConvergedReason *reason)
{
  PetscErrorCode ierr;

  ierr = SNESConvergedDefault(snes,it,xnorm,pnorm,fnorm,reason,NULL); CHKERRQ(ierr);

  // se não convergiu, não terminou ou não é um método ale, retorna
  if (*reason<=0 || !ale)
  {
    return ierr;
  }
  else
  {
    // se não é a primeira vez que converge
    if (converged_times)
    {
      return ierr;
    }
    else
    {

      //Vec *Vec_up_k = &Vec_up_1;
      ////SNESGetSolution(snes,Vec_up_k);
      //
      //copyMesh2Vec(Vec_x_1);
      //calcMeshVelocity(*Vec_up_k, Vec_x_1, Vec_v_mid, current_time+dt);
      //moveMesh(Vec_x_1, Vec_vmsh_0, Vec_v_mid, 0.5);
      //
      //// mean mesh velocity
      //VecAXPY(Vec_v_mid,1,Vec_vmsh_0);
      //VecScale(Vec_v_mid, 0.5);
      //
      //*reason = SNES_CONVERGED_ITERATING;
      //++converged_times;

      return ierr;
    }
  }


/*
  converged
  SNES_CONVERGED_FNORM_ABS         =  2,  ||F|| < atol
  SNES_CONVERGED_FNORM_RELATIVE    =  3,  ||F|| < rtol*||F_initial||
  SNES_CONVERGED_PNORM_RELATIVE    =  4,  Newton computed step size small; || delta x || < stol
  SNES_CONVERGED_ITS               =  5,  maximum iterations reached
  SNES_CONVERGED_TR_DELTA          =  7,
   diverged
  SNES_DIVERGED_FUNCTION_DOMAIN    = -1,  the new x location passed the function is not in the domain of F
  SNES_DIVERGED_FUNCTION_COUNT     = -2,
  SNES_DIVERGED_LINEAR_SOLVE       = -3,  the linear solve failed
  SNES_DIVERGED_FNORM_NAN          = -4,
  SNES_DIVERGED_MAX_IT             = -5,
  SNES_DIVERGED_LINE_SEARCH        = -6,  the line search failed
  SNES_DIVERGED_LOCAL_MIN          = -8,  || J^T b || is small, implies converged to local minimum of F()
  SNES_CONVERGED_ITERATING         =  0
*/
}

PetscErrorCode AppCtx::setUPInitialGuess()
{
  // set U^{n+1} b.c.

  VectorXi    u_dofs_fs(dim), p_dofs(dim);
  VectorXi    x_dofs(dim);
  int         tag;
  Vector      X1(dim);
  Vector      U1(dim);

  if (read_from_sv_fd){
    importSurfaceInfo(time_step);
  }

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    tag = point->getTag();

    getNodeDofs(&*point, DH_MESH, VAR_M, x_dofs.data());
    getNodeDofs(&*point, DH_UNKM, VAR_U, u_dofs_fs.data());

    VecGetValues(Vec_x_1, dim, x_dofs.data(), X1.data());

    if (is_in(tag, dirichlet_tags))
    {
      U1 = u_exact(X1, current_time+dt, tag);  //current_time+dt 'cause we want U^{n+1}
      VecSetValues(Vec_ups_1, dim, u_dofs_fs.data(), U1.data(), INSERT_VALUES);
    }
    else if (is_in(tag, solid_tags) || is_in(tag, feature_tags) || is_in(tag, triple_tags))
    {
      U1.setZero();
      VecSetValues(Vec_ups_1, dim, u_dofs_fs.data(), U1.data(), INSERT_VALUES);
    }
    else if (is_in(tag, slipvel_tags))
    {
      int nod_vs = is_in_id(tag,slipvel_tags);
      Vector Nr(dim), Vs(dim), Tg(dim), Bn(dim);
      VecGetValues(Vec_normal, dim, x_dofs.data(), Nr.data());
      int pID = mesh->getPointId(&*point);
      if (read_from_sv_fd){
        Vs = BFields_from_file(pID,2);
      }
      else{
        //Vs = SlipVel(X1, XG_0[nod_vs-1], Nr, dim, tag, theta_ini[nod_vs-1], 0.0, 0.0, current_time+unsteady*dt);//regular code
        Vs = SlipVel(X1, XG_1[nod_vs-1], Nr, dim, tag, theta_1[nod_vs-1], Kforp, nforp, current_time+unsteady*dt,
                     Q_1[nod_vs-1], thetaDOF);//for paramecium test
      }
      VecSetValues(Vec_slipv_1, dim, x_dofs.data(), Vs.data(), INSERT_VALUES);
      VecSetValues(Vec_slipv_0, dim, x_dofs.data(), Vs.data(), INSERT_VALUES);
      getTangents(Tg,Bn,Nr,dim);
      VecSetValues(Vec_tangent, dim, x_dofs.data(), Tg.data(), INSERT_VALUES);
    }
    else if (is_in(tag, flusoli_tags))
    {
      int nod_vs = is_in_id(tag,flusoli_tags);
      Vector Nr(dim), Ft(dim), Tg(dim), Bn(dim), Vm(dim);
      VecGetValues(Vec_normal, dim, x_dofs.data(), Nr.data());
      int pID = mesh->getPointId(&*point);
      if (read_from_sv_fd){
        Ft = BFields_from_file(pID,1);
      }
      else{
        Ft = FtauForce(X1, XG_1[nod_vs-1], Nr, dim, tag, theta_1[nod_vs-1], Kforp, nforp, current_time+unsteady*dt,
                       0*Nr, Q_1[nod_vs-1], thetaDOF, Kcte);//for paramecium test
      }
      VecSetValues(Vec_ftau_0, dim, x_dofs.data(), Ft.data(), INSERT_VALUES);
      getTangents(Tg,Bn,Nr,dim);
      VecSetValues(Vec_tangent, dim, x_dofs.data(), Tg.data(), INSERT_VALUES);
      Vm = SlipVel(X1, XG_1[nod_vs-1], Nr, dim, tag, theta_1[nod_vs-1], Kforp, nforp, current_time+unsteady*dt,
                   Q_1[nod_vs-1], thetaDOF);//for paramecium test
      VecSetValues(Vec_metav_0, dim, x_dofs.data(), Vm.data(), INSERT_VALUES);
    }

  } // end for point

  if (is_sflp){
    for (int nl = 0; nl < n_links; nl++){
      int dofs_sl = n_unknowns_u + n_unknowns_p + n_unknowns_z + nl;
      double link_vel = DFlink(current_time,nl);
      VecSetValue(Vec_ups_1, dofs_sl, link_vel, INSERT_VALUES);
    }
  }

  Assembly(Vec_ups_1);

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::setInitialConditions()
{
  PetscErrorCode      ierr(0);

  Vector    Uf(dim), Zf(LZ), Vs(dim), Nr(dim), Tg(dim), Bn(dim), Ftau(dim);
  Vector    X(dim);
  Tensor    R(dim,dim);
  VectorXi  dofs(dim), dofs_mesh(dim);  //global components unknowns enumeration from point dofs
  VectorXi  dofs_fs(LZ);
  Vector3d  Xg, XG_temp, Us, eref;
  int       nod_id, nod_is, tag, nod_vs, nodsum, dofs_sl;
  double    p_in, link_vel;
  vector<bool> SV(n_solids,false);  //solid visited history
  VectorXd  dllink(n_links);

  // Zero Entries ////////////////////////////////////////////////////////////////////////////////////////////////////
  VecZeroEntries(Vec_res_fs); //this size is the [U,P,S] sol vec size
  VecZeroEntries(Vec_res_m);
  VecZeroEntries(Vec_v_mid);  //this size(V) = size(X) = size(U)
  VecZeroEntries(Vec_v_1);
  VecZeroEntries(Vec_x_0);
  VecZeroEntries(Vec_x_1);
  VecZeroEntries(Vec_normal);
  VecZeroEntries(Vec_tangent);
  VecZeroEntries(Vec_ups_0);
  VecZeroEntries(Vec_ups_1);
  VecZeroEntries(Vec_ups_m1);
  VecZeroEntries(Vec_x_aux);
  if (is_bdf3){
    VecZeroEntries(Vec_ups_m2);
  }
  if (is_sfip){
    VecZeroEntries(Vec_slipv_0);
    VecZeroEntries(Vec_slipv_1);
    VecZeroEntries(Vec_slipv_m1);
    if (is_bdf3){VecZeroEntries(Vec_slipv_m2);}
    VecZeroEntries(Vec_res_Fdis);
    VecZeroEntries(Vec_fdis_0);
    VecZeroEntries(Vec_ftau_0);
    VecZeroEntries(Vec_metav_0);
  }
  if (is_sfip && (is_bdf2 || is_bdf3)){VecZeroEntries(Vec_x_cur);}
  if (is_sslv){
    VecZeroEntries(Vec_res_s);
    VecZeroEntries(Vec_slip_rho);
    VecZeroEntries(Vec_normal_aux);
    VecZeroEntries(Vec_rho_aux);
  }
  if (time_adapt){
    VecZeroEntries(Vec_ups_time_aux);
    VecZeroEntries(Vec_x_time_aux);
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  //if (dim == 3)
  //  VecZeroEntries(Vec_binormal);

  //////////////////////////////////////////////////
  copyMesh2Vec(Vec_x_0);  //copy initial mesh coordinates to Vec_x_0 = (a1,a2,a3,b1,b2,b3,c1,c2,c3,...)
  copyMesh2Vec(Vec_x_1);  //copy initial mesh coordinates to Vec_x_1
  // normals at boundary points (n01,n02,n03, n11,n12,n13, n21,n22,n23, ..., 0) following node order .geo
  getVecNormals(&Vec_x_0, Vec_normal);  //std::cout << VecView(Vec_normal,PETSC_VIEWER_STDOUT_WORLD)
  //Assembly(Vec_normal); View(Vec_normal,"matrizes/nr0.m","n0"); View(Vec_x_0,"matrizes/vx0.m","vxv0");//VecView(Vec_ups_0,PETSC_VIEWER_STDOUT_WORLD);

  // compute mesh sizes //////////////////////////////////////////////////
  getMeshSizes();
  //////////////////////////////////////////////////

  // calculate external slip velocity //////////////////////////////////////////////////
  if (is_sslv){
    calcSlipVelocity(Vec_x_1, Vec_slipv_0);
    //View(Vec_slipv_0,"matrizes/slv0.m","sl0");
  }//////////////////////////////////////////////////

  // setting initial rotational matrix for solids (identity) //////////////////////////////////////////////////
  if (is_sfip){
    Matrix3d Id3(Matrix3d::Identity(3,3));
    //bool dimt = dim == 3;
    //Id3 = dimt*Id3; //cout << Id3 << endl;
    for (int i = 0; i < n_solids; i++){
      Q_0.push_back(Id3); Q_1.push_back(Id3); Q_m1.push_back(Id3); Q_ini.push_back(Id3);
    }
    Q_0.resize(n_solids); Q_1.resize(n_solids); Q_m1.resize(n_solids); Q_ini.resize(n_solids);
  }//////////////////////////////////////////////////

  // calculate derivatives for link problem //////////////////////////////////////////////////
  if (is_sflp){
    //eref(0) = XG_ini[1](0) - XG_ini[0](0);
    //eref(1) = XG_ini[1](1) - XG_ini[0](1);
    for (int nl = 0; nl < n_links; nl++){
      eref = XG_0[nl+1] - XG_0[nl];
      eref.normalize();
      ebref.push_back(eref); //cout << ebref[nl].transpose() << endl;

      link_vel = DFlink(current_time, nl);
      //dllink.push_back(link_vel);
      dllink(nl) = link_vel;
      //cout << Flink(current_time, nl) << ",   " << DFlink(current_time, nl) << endl;
    }
    //dllink.resize(n_links);
    ebref.resize(n_links);
  }//////////////////////////////////////////////////

  // import data of slip vel, tangent forces, ..., from file //////////////////////////////////////////////////
  if (read_from_sv_fd){
    importSurfaceInfo(time_step);
  }//////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // initial velocity and initial pressure, saved in Vec_ups_0 (initial shoot) //////////////////////////////////////////////////
  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    point->getCoord(X.data(),dim);
    tag = point->getTag();
    // press //////////////////////////////////////////////////
    if (mesh->isVertex(&*point))
    {
      getNodeDofs(&*point, DH_UNKM, VAR_P, dofs.data());
      p_in = p_initial(X, tag);
      VecSetValues(Vec_ups_0,1,dofs.data(),&p_in,INSERT_VALUES);
      //VecSetValues(Vec_ups_1,1,dofs.data(),&p_in,INSERT_VALUES);
    }
    // vel //////////////////////////////////////////////////
    getNodeDofs(&*point, DH_UNKM, VAR_U, dofs.data());
    nod_id = is_in_id(tag,flusoli_tags);
    nod_is = is_in_id(tag,solidonly_tags);
    nod_vs = is_in_id(tag,slipvel_tags);
    nodsum = nod_id+nod_is+nod_vs;
    if (nodsum){//initial conditions coming from solid bodies//////////////////////////////////////////////////
      if (nod_is){//////////////////////////////////////////////////
        Zf = s_initial(dim, tag, LZ);  //first part of dofs vector q, in the case is_sflp this has the dofs of the referential body
        if (is_sflp){
          //cout << Zf.transpose() << endl;
          Zf = LinksVel(XG_0[nodsum-1], XG_0[0], theta_0[0], Q_0[0], Zf, dllink, nodsum, ebref/*[0]*/, dim, LZ);
          //cout << Zf.transpose() << endl;
        }
        //Uf = SolidVel(X, XG_0[nodsum-1], Zf, dim);//, is_sflp, theta_0[nodsum-1], dllink, nodsum, ebref[0]);
        Uf = SolidVelGen(X, XG_0[nodsum-1], theta_0[0], modes_0[0], Zf, dim, is_axis);
        VecSetValues(Vec_ups_0, dim, dofs.data(), Uf.data(), INSERT_VALUES);
      }//////////////////////////////////////////////////
      if (nod_vs+nod_id){ ////////////////////////////////////////////////// //ojo antes solo nod_vs
        getNodeDofs(&*point, DH_MESH, VAR_M, dofs_mesh.data());
        int pID = mesh->getPointId(&*point);
        if(!is_sslv){  //calculate slip velocity at slipvel nodes (and in fsi nodes, which is not a problem, cause this is just a shoot) TODO
          VecGetValues(Vec_normal, dim, dofs_mesh.data(), Nr.data());
          /*cout << X.transpose() << "   "
               << XG_0[nod_vs+nod_id-1].transpose() << "   " << Nr.transpose() << "   " << theta_ini[nod_vs+nod_id-1] << endl;*/
          if (nod_vs){
            if (read_from_sv_fd){
              Vs = BFields_from_file(pID,2);
            }
            else{
              //Vs = SlipVel(X, XG_0[nod_vs+nod_id-1], Nr, dim, tag, theta_ini[nod_vs+nod_id-1], 0.0, 0.0, current_time);  //ojo antes solo nod_vs//here nod_vs=0
              Vs = SlipVel(X, XG_0[nod_vs-1], Nr, dim, tag, theta_0[nod_vs-1], Kforp, nforp, current_time+unsteady*dt,
                           Q_0[nod_vs-1], thetaDOF);//for paramecium test
            }
            VecSetValues(Vec_slipv_0, dim, dofs_mesh.data(), Vs.data(), INSERT_VALUES);
          }
          else if (nod_id){
            if (read_from_sv_fd){
              Ftau = BFields_from_file(pID,1);  //cout << pID << " " << Ftau.transpose() << endl;
            }
            else{
              Vs.setZero();
              Ftau = FtauForce(X, XG_0[nod_vs+nod_id-1], Nr, dim, tag, theta_0[nod_vs+nod_id-1], Kforp, nforp, current_time+unsteady*dt,
                               Vs, Q_0[nod_vs+nod_id-1], thetaDOF, Kcte);
            }
            VecSetValues(Vec_ftau_0, dim, dofs_mesh.data(), Ftau.data(), INSERT_VALUES);
            Vs = SlipVel(X, XG_0[nod_vs-1], Nr, dim, tag, theta_0[nod_vs-1], Kforp, nforp, current_time+unsteady*dt,
                         Q_0[nod_vs-1], thetaDOF);//for paramecium test
            VecSetValues(Vec_metav_0, dim, dofs_mesh.data(), Vs.data(), INSERT_VALUES);
          }
          getTangents(Tg,Bn,Nr,dim);  //cout << Nr.transpose() << " * " << Tg.transpose() << " = " << Nr.dot(Tg) << " Bin " << Bn.transpose() << endl;
          VecSetValues(Vec_tangent, dim, dofs_mesh.data(), Tg.data(), INSERT_VALUES);
          //if (dim == 3)
          //  VecSetValues(Vec_binormal, dim, dofs_mesh.data(), Bn.data(), INSERT_VALUES);
        }
        else
        {
          VecGetValues(Vec_slipv_0, dim, dofs_mesh.data(), Vs.data());
        }
        //Uf = Uf + Vs;  //cout << tag << "  " << X(0)-3 << " " << X(1)-3<< "  " << Uf.transpose() << endl; ojo, descomentar?
      }
    }//end body nodes//////////////////////////////////////////////////

    else{//fluid nodes//////////////////////////////////////////////////
      Uf = u_initial(X, tag);
      VecSetValues(Vec_ups_0, dim, dofs.data(), Uf.data(), INSERT_VALUES);  //cout << dofs.transpose() << endl;
    }//////////////////////////////////////////////////
  }// end point loop //////////////////////////////////////////////////

  if (is_sfip){//calculating the first interpolated/approximated value for the body's DOFs //////////////////////////////////////////////////
    for (int K = 1; K <= n_solids ; K++){
      nodsum = K;
      for (int l = 0; l < LZ; l++){
        dofs_fs(l) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + l;
      }
      Zf = s_initial(dim, solidonly_tags[nodsum-1], LZ);
      VecSetValues(Vec_ups_0, LZ, dofs_fs.data(), Zf.data(), INSERT_VALUES);  //saving DOFs in the UPS vector//cout << dofs_fs.transpose() << endl;
/*      Xg = XG_0[nodsum-1];
      Xg(0) = Xg(0)+dt*Zf(0); Xg(1) = Xg(1)+dt*Zf(1); if (dim == 3){Xg(2) = Xg(2)+dt*Zf(2);}
      XG_1[nodsum-1] = Xg;                               // if Zf = 0, then XG_1 = XG_0
      theta_1[nodsum-1] = theta_0[nodsum-1] + dt*Zf(2);  // if Zf = 0, then theta_1 = theta_0
      //Q_1[nodsum-1] = Q_0[nodsum-1] + 0*dt*SkewMatrix(Zf.tail(3),dim);  //TODO: matrix ode solver*/
    }
  }////////////////////////////////////////////////////////////////////////////////////////////////////

  if (is_sflp){//calculating the links derivatives and saving it in UPS //////////////////////////////////////////////////
    for (int nl = 0; nl < n_links; nl++){
      dofs_sl = n_unknowns_u + n_unknowns_p + n_unknowns_z + nl;
      link_vel = DFlink(current_time,nl);  //cout << link_vel << endl;
      VecSetValue(Vec_ups_0, dofs_sl, link_vel, INSERT_VALUES);
    }
  }////////////////////////////////////////////////////////////////////////////////////////////////////

  if (false){//saving matlab matrices //////////////////////////////////////////////////
    View(Vec_x_0,"matrizes/xv0.m","x0");
    if (is_sfip){View(Vec_slipv_0,"matrizes/sv0.m","s0");}
    View(Vec_tangent,"matrizes/tv0.m","t0");
    View(Vec_normal,"matrizes/nr0.m","n0");
    Assembly(Vec_ups_0);  View(Vec_ups_0,"matrizes/vuzp0.m","vuzp0m");//VecView(Vec_ups_0,PETSC_VIEWER_STDOUT_WORLD);
    if (true) {getFromBSV();}  //to calculate the analytical squirmer velocity
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Assembly(Vec_slipv_0);  View(Vec_slipv_0,"matrizes/slip0.m","slipm");//VecView(Vec_ups_0,PETSC_VIEWER_STDOUT_WORLD);
  VecCopy(Vec_ups_0,Vec_ups_1);  //PetscInt size1; //u_unk+z_unk+p_unk //VecGetSize(Vec_up_0,&size1);
  //VecCopy(Vec_ups_0,Vec_ups_m1); //This to save the uzp_0 temporarily in uzp_m1
  VecCopy(Vec_slipv_0, Vec_slipv_1);

  //Plot initial conditions, initial mesh velocity = 0, initial slip velocity distribution//////////////////////////////////////////////////
  if (family_files){plotFiles(0);}
  //Print mech. dofs info
  sprintf(grac,"%s/HistGra.txt",filehist_out.c_str());
  sprintf(velc,"%s/HistVel.txt",filehist_out.c_str());
  sprintf(errc,"%s/HistErr.txt",filehist_out.c_str());
  sprintf(rotc,"%s/HistRot.txt",filehist_out.c_str());
  sprintf(forc,"%s/HistFor.txt",filehist_out.c_str());
  if (fprint_hgv){
    filg.open(grac); filv.open(velc); filr.open(errc); filt.open(rotc); filf.open(forc);
    filg.close();    filv.close();    filr.close();    filt.close();    filf.close();
  }
  saveDOFSinfo(0);
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  cout << "\n==================================================\n";
  cout << "current time: " << current_time << endl;
  cout << "time step: "    << time_step    << endl;
  //cout << "steady error: " << steady_error << endl;
  cout << "--------------------------------------------------";
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // Picard iterations (predictor-corrector to initialize) //////////////////////////////////////////////////
  for (int pic = 0; pic < PI; pic++)
  {
    printf("\n\tFixed Point Iteration (Picard) %d\n", pic);

    if (!static_mesh && time_step > 0){//////////////////////////////////////////////////
      //Extrapolation/advance of solid's DOFs//////////////////////////////////////////////////
      if (is_sfip){//fluid-solid case//////////
        moveSolidDOFs(stheta);  //stheta = 1/2, order 2
      }
      else{//fliud-fluid case//////////
        VecWAXPY(Vec_x_1, dt/2.0, Vec_v_mid, Vec_x_0); // Vec_x_1 = dt*Vec_v_mid + Vec_x_0 // for zero Dir. cond. solution lin. elast. is Vec_v_mid = 0
      }//////////////////////////////////////////////////

      //Mesh compatibilization, elasticity problem and slip vel at extrap mesh//////////////////////////////////////////////////
      calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 1.0, Vec_v_mid, 0.0);
      VecWAXPY(Vec_x_1, dt, Vec_v_mid, Vec_x_0);  //FIXME: false for Re=0 static mesh
/*
      //For Re advancing//////////////////////////////////////////////////
      if (false){
        VectorXd v_coeffs_s(LZ);
        for (int l = 0; l < LZ; l++){
          dofs_fs(l) = n_unknowns_u + n_unknowns_p + l;
        }
        VecGetValues(Vec_ups_1,dofs_fs.size(),dofs_fs.data(),v_coeffs_s.data());
        //if (pic == 0){v_coeffs_s(1) = -0.315364702750772;}
        VecScale(Vec_v_mid, v_coeffs_s(1));
        //View(Vec_v_mid,"matrizes/vmid0.m","vm");
      }////////////////////////////////////////////////////////////////////////////////////////////////////
*/

      // extrapolation and compatibilization of the mesh
      //VecWAXPY(Vec_x_1, dt/2.0, Vec_v_mid, Vec_x_0); // Vec_x_1 = dt*Vec_v_mid + Vec_x_0 // for zero Dir. cond. solution lin. elast. is Vec_v_mid = 0
      //if (is_sfip){updateSolidMesh();} //extrap of mech. system dofs, and compatibilization of mesh through the mesh vel.

      //Mesh adaptation (it has topological changes) (2d only) it destroys Vec_normal//////////////////////////////////////////////////
      if (mesh_adapt){if (is_sfim){meshAdapt_d();} else {meshAdapt_s();}}
      copyVec2Mesh(Vec_x_1);
      if (mesh_adapt){meshFlipping_s();}
      if (true) {CheckInvertedElements();}  //true just for mesh tests FIXME
    }//end moving the mesh//////////////////////////////////////////////////

    //Calculate normals//////////////////////////////////////////////////
    getVecNormals(&Vec_x_1, Vec_normal);
    //////////////////////////////////////////////////

    //Initial guess for non-linear solver//////////////////////////////////////////////////
    setUPInitialGuess();  // initial guess saved at Vec_ups_1
    //////////////////////////////////////////////////

    // * SOLVE THE SYSTEM * //////////////////////////////////////////////////
    if (solve_the_sys)
    { //VecView(Vec_ups_1,PETSC_VIEWER_STDOUT_WORLD); //VecView(Vec_ups_0,PETSC_VIEWER_STDOUT_WORLD);
      cout << "-----System solver-----" << endl;
      ierr = SNESSolve(snes_fs,PETSC_NULL,Vec_ups_1);  CHKERRQ(ierr); Assembly(Vec_ups_1);  View(Vec_ups_1,"matrizes/vuzp1.m","vuzp1m");
      cout << "--------------------------------------------------" << endl;
    }////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    if (is_sfip){// update mesh and mech. dofs//////////////////////////////////////////////////
      if (is_mr_qextrap){
        moveSolidDOFs(1.0);
        calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 1.0, Vec_v_mid, 0.0);
        VecWAXPY(Vec_x_1, dt, Vec_v_mid, Vec_x_0);
      }
    }////////////////////////////////////////////////////////////////////////////////////////////////////
*/
/*
    //For Re advancing//////////////////////////////////////////////////
    if (false){
      if (pic != 0 && pic%3 == 0){
        saveDOFSinfo_Re_Vel();
        if (family_files){plotFiles(1);}
        //computeForces(Vec_x_0,Vec_ups_1);
        computeViscousDissipation(Vec_x_0,Vec_ups_1);
        computeError(Vec_x_0,Vec_ups_1,current_time);
        time_step += 1;
      }
    }////////////////////////////////////////////////////////////////////////////////////////////////////
*/

    //For virtual only time advancing (i.e. not moving mesh)////////////////////////////
    if (static_mesh){
      //save "velocity at infinity" constant vector field in vec_v_mid
      VectorXd v_coeffs_s(LZ);
      for (int l = 0; l < LZ; l++){
        dofs_fs(l) = n_unknowns_u + n_unknowns_p + l;
      }
      calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 1.0, Vec_v_mid, 0.0);
      VecGetValues(Vec_ups_1,dofs_fs.size(),dofs_fs.data(),v_coeffs_s.data());
      //if (pic == 0){v_coeffs_s(1) = -0.315364702750772;}
      VecScale(Vec_v_mid, max(v_coeffs_s(1),v_coeffs_s(0)));
      //View(Vec_v_mid,"matrizes/vmid0.m","vm");
      ////////////////////////////////////////////////////////////////////////////////////////////////////
      //calculate interaction force and prepare the slipvel, ftauforce and Fdforce to be printed///////////////
      if (true && is_sfip && ((flusoli_tags.size() != 0)||(slipvel_tags.size() != 0)) /*&& (pic+1 == PI)*/){
        cout << "-----Interaction force calculation------" << endl;
        ierr = SNESSolve(snes_fd,PETSC_NULL,Vec_fdis_0);  CHKERRQ(ierr);
        cout << "-----Interaction force extracted------" << endl;
        extractForce(true);
      }
      // save data for current time ///////////////////////////////////////////////////////////////////
      getSolidVolume();
      saveDOFSinfo(0);
      if (family_files){plotFiles(1);}
      //computeForces(Vec_x_1,Vec_ups_1);
      computeViscousDissipation(Vec_x_1,Vec_ups_1);
      computeError(Vec_x_1,Vec_ups_1,current_time);
      current_time += dt;
      time_step += 1;
      cout << "\n==================================================\n";
      cout << "current time: " << current_time << endl;
      cout << "time step: "    << time_step  << endl;
      //cout << "steady error: " << steady_error << endl;
      cout << "--------------------------------------------------";
      if (pic == PI-1){
        cout << "\n--------------------------------------------------";
        cout << "\nFinishg Picard Iterations or Stokes time advancing\n";
        cout << "--------------------------------------------------\n";
        current_time -= dt;
        time_step -= 1;
      }
    }////////////////////////////////////////////////////////////////////////////////////////////////////
    //For regular Stokes moving mesh//////////////////////////////////////////////////
    else {
      if (time_step == 0){//this implies that the zero time solution is calculated instantly
                          //with the sufficient info known at this time, no need to make picard iterations
        //calculate interaction force and prepare the slipvel, ftauforce and Fdforce to be printed///////////////
        if (false && is_sfip && ((flusoli_tags.size() != 0)||(slipvel_tags.size() != 0)) /*&& (pic+1 == PI)*/){
          cout << "-----Interaction force calculation------" << endl;
          ierr = SNESSolve(snes_fd,PETSC_NULL,Vec_fdis_0);  CHKERRQ(ierr);
          cout << "-----Interaction force extracted------" << endl;
          extractForce(false);
        }
        //save data for current time///////////////////////////////////////////////////////////////////
        getSolidVolume();
        saveDOFSinfo(0);
        if (family_files){plotFiles(1);}
        //computeForces(Vec_x_1,Vec_ups_1);
        computeViscousDissipation(Vec_x_1,Vec_ups_1);
        computeError(Vec_x_1,Vec_ups_1,current_time);
        current_time += dt;
        time_step += 1;
        cout << "\n==================================================\n";
        cout << "current time: " << current_time << endl;
        cout << "time step: "    << time_step  << endl;
        //cout << "steady error: " << steady_error << endl;
        cout << "--------------------------------------------------";
        pic = pic-1;  //this implies that time_step=1 will be solved with complete picard iterations

        //Prepare data for time_step=1///////////////////////////////////////////////////////////////////
        VecCopy(Vec_ups_1,Vec_ups_0);//In Stokes the previously Vec_ups_0 has no important info at all
        ////////////////////////////////////////////////////////////////////////////////////////////////////
      }
    }////////////////////////////////////////////////////////////////////////////////////////////////////


  }// end Picard Iterartions loop //////////////////////////////////////////////////
  //copyMesh2Vec(Vec_x_0); // at this point X^{0} is the original mesh, and X^{1} the next mesh

  if (static_mesh){//////////////////////////////////////////////////
    cout << "--------------------------------------------------" << endl;
  }
  else{//////////////////////////////////////////////////
    // calculate interaction force and prepare the slipvel, ftauforce and Fdforce to be printed///////////////
    if (false && is_sfip && ((flusoli_tags.size() != 0)||(slipvel_tags.size() != 0)) /*&& (pic+1 == PI)*/){
      cout << "-----Interaction force calculation------" << endl;
      ierr = SNESSolve(snes_fd,PETSC_NULL,Vec_fdis_0);  CHKERRQ(ierr);
      cout << "-----Interaction force extracted------" << endl;
      extractForce(false);
    }
    // save data for current time ///////////////////////////////////////////////////////////////////
    getSolidVolume();
    saveDOFSinfo(1);
    if (family_files){plotFiles(1);}
  }//////////////////////////////////////////////////
  //computeForces(Vec_x_1,Vec_ups_1);
  computeViscousDissipation(Vec_x_1,Vec_ups_1);
  computeError(Vec_x_1,Vec_ups_1,current_time);

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::solveTimeProblem()
{
  PetscErrorCode      ierr(0);

  double   initial_volume = getMeshVolume(), final_volume, Qmesh = quality_m(&*mesh);
  Vector   X(dim), Xe(dim), U0(Vector::Zero(3)), U1(Vector::Zero(3)), XG_temp(3);
  double   x_error=0;
  VectorXi dofs(dim);

  if (false && print_to_matlab)
    printMatlabLoader();

  if (is_sfip){
    if (true && !is_axis){
      getSolidVolume();
      //getSolidCentroid();
    }
    getSolidInertiaTensor();
  }
  cout << endl;
  cout.precision(15);
  for (int K = 0; K < n_solids; K++){
    cout << K+1 << ": volume = " << VV[K] <<  "; center of mass = " << XG_0[K].transpose() << "; inertia tensor = " << endl;
    cout << InTen[K] << endl;
  }
  cout << endl;

  SarclTest();

  // ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Solve nonlinear system
  // ////////////////////////////////////////////////////////////////////////////////////////////////////
  printf("Initial volume: %.15lf\n", initial_volume);
  printf("Initial quality mesh: %.15lf\n", Qmesh);
  printf("Num. of time iterations (maxts): %d\n",maxts);
  printf("Starting time loop... \n\n");

  current_time = 0;
  time_step = 0;

  double Qmax = 0;
  double steady_error = 1;

  //Solving the 0 and 1 time_step//////////////////////////////////////////////////
  setInitialConditions();  //called only here
  /////////////////////////////////////////////////////////////////////////////////

  int its;

/*
  else if (is_bdf2)
  {
    current_time += dt;
    time_step += 1;

    VecCopy(Vec_ups_0,Vec_ups_m1);
    copyVec2Mesh(Vec_x_1);

    if (is_bdf2_bdfe)
    {
      // extrapolation \tilde{X}^{n+1}=2X^{n}-X^{n-1}
      VecScale(Vec_x_1, 2.0);
      VecAXPY(Vec_x_1,-1.0,Vec_x_0);
      copyMesh2Vec(Vec_x_cur);
      // calc V^{n+1} and update with D_{2}X^{n+1} = dt*V^{n+1}
      if (is_sfip){
        moveSolidDOFs(2.0);
        if (is_slipv) VecCopy(Vec_slipv_1,Vec_slipv_0);
      }
      calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 2.0, Vec_v_1, current_time);
      VecCopy(Vec_v_1, Vec_x_1);
      VecScale(Vec_x_1, 2./3.*dt);
      VecAXPY(Vec_x_1,-1./3.,Vec_x_0);
      copyMesh2Vec(Vec_x_0);
      VecAXPY(Vec_x_1,4./3.,Vec_x_0);
      if (is_sfip && false){
        moveSolidDOFs(2.0);
        if (is_slipv) VecCopy(Vec_slipv_1,Vec_slipv_0);
        updateSolidMesh();
      }
      VecCopy(Vec_v_1,Vec_v_mid);
    }
    else if (is_bdf2_ab)
    {
      // extrapolated geometry
      //VecScale(Vec_x_1, 2.0);
      //VecAXPY(Vec_x_1,-1.0,Vec_x_0);  // \bar{X}^(n+1/2)=2.0*X^(n)-1.0X^(n-1)
      VecScale(Vec_x_1, 1.5);
      VecAXPY(Vec_x_1,-0.5,Vec_x_0);  // \bar{X}^(n+1/2)=1.5*X^(n)-0.5X^(n-1)
      copyMesh2Vec(Vec_x_0);          //copy current mesh to Vec_x_0
      if (is_sfip){
        moveSolidDOFs(1.5);
        if (is_slipv) VecCopy(Vec_slipv_1,Vec_slipv_0);
      }
      calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 1.5, Vec_v_1, current_time); // Adams-Bashforth
      VecWAXPY(Vec_x_1, dt, Vec_v_1, Vec_x_0); // Vec_x_1 = Vec_v_1*dt + Vec_x_0
      if (is_sfip && false){
        moveSolidDOFs(1.5);
        if (is_slipv) VecCopy(Vec_slipv_1,Vec_slipv_0);
        updateSolidMesh();
      }
      // velocity at integer step
      //VecScale(Vec_v_1, 1.5);
      //VecAXPY(Vec_v_1,-.5,Vec_v_mid);
      VecScale(Vec_v_1, 2.0);
      VecAXPY(Vec_v_1,-1.0,Vec_v_mid);
    }

    VecCopy(Vec_ups_1, Vec_ups_0);
    if (is_sslv){
      calcSlipVelocity(Vec_x_1, Vec_slipv_1);
    }
  }
  else if (is_mr_ab){
    current_time += dt;
    time_step += 1;
    copyVec2Mesh(Vec_x_1);
    // extrapolated geometry
    VecScale(Vec_x_1, 2.0);         // look inside residue why this extrap is used
    VecAXPY(Vec_x_1,-1.0,Vec_x_0);  // \bar{X}^(n+1/2)=2.0*X^(n)-1.0X^(n-1)
    //VecScale(Vec_x_1, 1.5);
    //VecAXPY(Vec_x_1,-0.5,Vec_x_0);  // \bar{X}^(n+1/2)=1.5*X^(n)-0.5X^(n-1)
    copyMesh2Vec(Vec_x_0);          //copy current mesh to Vec_x_0
    //velNoSlip(Vec_ups_0,Vec_slipv_0,Vec_ups_0_ns);velNoSlip(Vec_ups_1,Vec_slipv_1,Vec_ups_1_ns);
    // extrapolate center of mass
    if (is_sfip){
      moveSolidDOFs(1.5);
      if (is_slipv) VecCopy(Vec_slipv_1,Vec_slipv_0);
    }
    calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 1.5, Vec_v_mid, current_time); // Adams-Bashforth
    VecWAXPY(Vec_x_1, dt, Vec_v_mid, Vec_x_0); // Vec_x_1 = Vec_v_mid*dt + Vec_x_0
    if (is_sslv){
      calcSlipVelocity(Vec_x_1, Vec_slipv_1);
    }
    VecCopy(Vec_ups_1, Vec_ups_0);
  }
*/


  //print solid's center information
  //ofstream filg, filv;
  //Vector3d Xgg;
  //VectorXd v_coeffs_s(LZ*n_solids);
  //VectorXi mapvs(LZ*n_solids);
  //int TT = 0;
  //char grac[PETSC_MAX_PATH_LEN], velc[PETSC_MAX_PATH_LEN];

  for(;;)  // equivalent to forever or while(true), must be a break inside
  {
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    current_time += dt;
    time_step += 1;
    cout << "\n==================================================\n";
    cout << "current time: " << current_time << endl;
    cout << "time step: "    << time_step  << endl;
    cout << "steady error: " << steady_error << endl;
    cout << "--------------------------------------------------";
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //Preparing data at time n//////////////////////////////////////////////////
    VecCopy(Vec_x_0, Vec_x_aux);
    VecCopy(Vec_x_1, Vec_x_0);
    XG_m1 = XG_0; theta_m1 = theta_0; Q_m1  = Q_0; modes_m1 = modes_0;
    XG_0  = XG_1; theta_0  = theta_1; Q_0   = Q_1; modes_0  = modes_1;
    VecCopy(Vec_slipv_1, Vec_slipv_0);
    VecCopy(Vec_ups_1,Vec_ups_m1);
    //copyVec2Mesh(Vec_x_1);
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Saving and printing data //////////////////////////////////////////////////
    //saveDOFSinfo();

/*    if ((is_basic || is_mr_qextrap) && !unsteady && time_step > 0 && maxts == 2){
      plotFiles();
      cout << "\n==================================================\n";
      cout << "stop reason:\n";
      cout << "maximum number of iterations reached. \n";
      break;
    }*/

    // Picard iterations (predictor-corrector to initialize) //////////////////////////////////////////////////
    for (int pic = 0; pic < PIs; pic++)
    {
      printf("\n\tFixed Point Iteration (Picard) %d\n", pic);

      //Extrapolation/advance of solid's DOFs//////////////////////////////////////////////////
      if (is_sfip){
        moveSolidDOFs(stheta);  //stheta = 1/2, order 2
      }
      else{//fliud-fluid case
        VecWAXPY(Vec_x_1, dt/2.0, Vec_v_mid, Vec_x_0); // Vec_x_1 = dt*Vec_v_mid + Vec_x_0 // for zero Dir. cond. solution lin. elast. is Vec_v_mid = 0
      }

      //Mesh compatibilization, elasticity problem and slip vel at extrap mesh//////////////////////////////////////////////////
      calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 1.0, Vec_v_mid, 0.0);
      VecWAXPY(Vec_x_1, dt, Vec_v_mid, Vec_x_0);

      // if found inverted element continue ///////////////////////////////////////////////////////////////////
      if (time_adapt){
        CheckInvertedElements();
        if (inverted_elem){
          break;
        }
      }////////////////////////////////////////////////////////////////////////////////////////////////////

      //Extrapolation and compatibilization of the mesh//////////////////////////////////////////////////
      //VecWAXPY(Vec_x_1, dt/2.0, Vec_v_mid, Vec_x_0); // Vec_x_1 = dt*Vec_v_mid + Vec_x_0 // for zero Dir. cond. solution lin. elast. is Vec_v_mid = 0
      //if (is_sfip){updateSolidMesh();} //extrap of mech. system dofs, and compatibilization, and slip vel at extrap mesh
      //copyVec2Mesh(Vec_x_1);  //not sure if necessary

      //Mesh adaptation (it has topological changes) (2d only) it destroys Vec_normal//////////////////////////////////////////////////
      if (mesh_adapt){if (is_sfim){meshAdapt_d();} else {meshAdapt_s();}}
      copyVec2Mesh(Vec_x_1);
      if (mesh_adapt){meshFlipping_s();}
      if (true) {CheckInvertedElements();}  //true just for mesh tests FIXME
      //////////////////////////////////////////////////

      //Calculate normals//////////////////////////////////////////////////
      getVecNormals(&Vec_x_1, Vec_normal);
      //////////////////////////////////////////////////

      //////////////////////////////////////////////////
      ////if ((pic+1) < PIs){
      //if (pic == 0){//copy n to n-1 just in the first Poincaré Iteration and save ups_0
      //  VecCopy(Vec_ups_1, Vec_ups_0);
      //  //if (family_files){plotFiles();}
      //}
      //}//////////////////////////////////////////////////

      //Initial guess for non-linear solver//////////////////////////////////////////////////
      setUPInitialGuess();  //setup Vec_ups_1 for SNESSolve
      //////////////////////////////////////////////////

      // * SOLVE THE SYSTEM * ///////////////////////////////////////////////////////////////////////////
      if (solve_the_sys){
        cout << "-----System solver-----" << endl;
        ierr = SNESSolve(snes_fs,PETSC_NULL,Vec_ups_1);  CHKERRQ(ierr); //Assembly(Vec_ups_1);  View(Vec_ups_1,"matrizes/vuzp1.m","vuzp1m");//VecView(Vec_ups_0,PETSC_VIEWER_STDOUT_WORLD);
        //ierr = SNESGetIterationNumber(snes_fs,&its);     CHKERRQ(ierr);
        //cout << "# snes iterations: " << its << endl
        cout << "--------------------------------------------------" << endl;
      }////////////////////////////////////////////////////////////////////////////////////////////////////

/*
      if (is_sfip){// update mesh and mech. dofs//////////////////////////////////////////////////
        if (is_mr_qextrap){
          moveSolidDOFs(1.0);
          calcMeshVelocity(Vec_x_0, Vec_ups_0, Vec_ups_1, 1.0, Vec_v_mid, 0.0);
          VecWAXPY(Vec_x_1, dt, Vec_v_mid, Vec_x_0);
        }
      }////////////////////////////////////////////////////////////////////////////////////////////////////
*/

    }// end Picard Iterartions loop //////////////////////////////////////////////////

    // if found inverted element continue ///////////////////////////////////////////////////////////////////
    if (time_adapt){
      if (inverted_elem){
        current_time -= dt;
        time_step -= 1;
        cout << "Inverted Element Found... Going back to previous time."<< "dt = " << dt << endl;
        dt = dt/2.0;
        VecCopy(Vec_x_0, Vec_x_1);
        XG_1 = XG_0; theta_1 = theta_0; Q_1 = Q_0;
        VecCopy(Vec_slipv_0, Vec_slipv_1);
        cin.get();
        continue;
      }
    }////////////////////////////////////////////////////////////////////////////////////////////////////

    // calculate interaction force and prepare the slipvel, ftauforce and Fdforce to be printed///////////////
    if (false && is_sfip && ((flusoli_tags.size() != 0)||(slipvel_tags.size() != 0)) /*&& (pic+1 == PIs)*/){
      cout << "-----Interaction force calculation-----" << endl;
      ierr = SNESSolve(snes_fd,PETSC_NULL,Vec_fdis_0);  CHKERRQ(ierr);
      cout << "-----Interaction force extracted------" << endl;
      extractForce(false);
    }
    // save data for current time ///////////////////////////////////////////////////////////////////
    getSolidVolume();
    saveDOFSinfo(1);
    if (family_files){plotFiles(1);}
    //computeForces(Vec_x_1,Vec_ups_1);
    computeViscousDissipation(Vec_x_1,Vec_ups_1);
    computeError(Vec_x_1,Vec_ups_1,current_time);
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //Prepare data for next time_step///////////////////////////////////////////////////////////////////
    VecCopy(Vec_ups_m1,Vec_ups_0);
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //cout << "--------------------------------------------------" << endl;
    // compute steady error //////////////////////////////////////////////////
    VecNorm(Vec_ups_1, NORM_1, &Qmax); //VecNorm(Vec_up_1, NORM_1, &Qmax);
    VecCopy(Vec_ups_0, Vec_res_fs); //VecCopy(Vec_up_0, Vec_res);
    VecAXPY(Vec_res_fs,-1.0,Vec_ups_1); //VecAXPY(Vec_res,-1.0,Vec_up_1);//steady_error = VecNorm(Vec_res, NORM_1)/(Qmax==0.?1.:Qmax);
    VecNorm(Vec_res_fs, NORM_1, &steady_error); //VecNorm(Vec_res, NORM_1, &steady_error);
    steady_error /= (Qmax==0.?1.:Qmax);

    if(time_step+1 > maxts) {
      cout << "\n==================================================\n";
      cout << "stop reason:\n";
      cout << "maximum number of iterations reached. \n";
      break;
    }
    if (steady_error <= steady_tol) {
      cout << "\n==================================================\n";
      cout << "stop reason:\n";
      cout << "steady state reached. \n";
      break;
    }
  }
  // ///////////////////////////////////////////////////////////////////////////
  // END TIME LOOP
  // ///////////////////////////////////////////////////////////////////////////

  //VecCopy(Vec_ups_1, Vec_ups_0);
  //if (family_files){plotFiles(1);}
  //cout << endl;

  final_volume = getMeshVolume();
  printf("final volume: %.15lf \n", final_volume);
  printf("volume error 100*(f-i)/i: %.15lf per percent\n", 100*abs(final_volume-initial_volume)/initial_volume);
  printf("x error : %.15lf \n", x_error);

  SNESConvergedReason reason;
  ierr = SNESGetIterationNumber(snes_fs,&its);     CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes_fs,&reason);  CHKERRQ(ierr);
  cout << "num snes iterations: " << its << endl;
  cout << "reason: " << reason << endl;
  switch (reason)
  {
  case( SNES_CONVERGED_FNORM_ABS       ): printf("SNES_CONVERGED_FNORM_ABS      /* ||F|| < atol */\n"); break;
  case( SNES_CONVERGED_FNORM_RELATIVE  ): printf("SNES_CONVERGED_FNORM_RELATIVE /* ||F|| < rtol*||F_initial|| */\n"); break;
  case( SNES_CONVERGED_SNORM_RELATIVE  ): printf("SNES_CONVERGED_PNORM_RELATIVE /* Newton computed step size small); break; || delta x || < stol */\n"); break;
  case( SNES_CONVERGED_ITS             ): printf("SNES_CONVERGED_ITS            /* maximum iterations reached */\n"); break;
  case( SNES_CONVERGED_TR_DELTA        ): printf("SNES_CONVERGED_TR_DELTA       \n"); break;
        /* diverged */
  case( SNES_DIVERGED_FUNCTION_DOMAIN  ): printf("SNES_DIVERGED_FUNCTION_DOMAIN /* the new x location passed the function is not in the domain of F */\n"); break;
  case( SNES_DIVERGED_FUNCTION_COUNT   ): printf("SNES_DIVERGED_FUNCTION_COUNT   \n"); break;
  case( SNES_DIVERGED_LINEAR_SOLVE     ): printf("SNES_DIVERGED_LINEAR_SOLVE    /* the linear solve failed */\n"); break;
  case( SNES_DIVERGED_FNORM_NAN        ): printf("SNES_DIVERGED_FNORM_NAN       \n"); break;
  case( SNES_DIVERGED_MAX_IT           ): printf("SNES_DIVERGED_MAX_IT          \n"); break;
  case( SNES_DIVERGED_LINE_SEARCH      ): printf("SNES_DIVERGED_LINE_SEARCH     /* the line search failed */ \n"); break;
  case( SNES_DIVERGED_LOCAL_MIN        ): printf("SNES_DIVERGED_LOCAL_MIN       /* || J^T b || is small, implies converged to local minimum of F() */\n"); break;
  case( SNES_CONVERGED_ITERATING       ): printf("SNES_CONVERGED_ITERATING      \n"); break;
  case( SNES_DIVERGED_INNER            ): printf("SNES_DIVERGED_INNER           \n"); break;
  }

  if (solve_the_sys)
    MatrixInfo(Mat_Jac_fs); //MatrixInfo(Mat_Jac);

  int lits;
  SNESGetLinearSolveIterations(snes_fs,&lits); //SNESGetLinearSolveIterations(snes,&lits);

  cout << "Greatest error reached during the simulation:" << endl;
  //printf("%-21s %-21s %-21s %-21s %-21s %-21s %-21s %-21s %s\n",
  //         "# hmean", "u_L2_norm", "p_L2_norm", "grad_u_L2_norm", "grad_p_L2_norm", "u_L2_facet_norm", "u_inf_facet_norm", "u_inf_norm", "p_inf_norm" );
  //printf("%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n\n",
  //         Stats.hmean, Stats.u_L2_norm, Stats.p_L2_norm, Stats.grad_u_L2_norm, Stats.grad_p_L2_norm, Stats.u_L2_facet_norm,  Stats.u_inf_facet_norm, Stats.u_inf_norm, Stats.p_inf_norm);
  printf("%-21s %-21s %-21s %-21s %-21s %-21s %-21s %-21s %-21s\n",
            "# hmean","u_L2_norm","p_L2_norm","u_inf_norm","p_inf_norm","u_L2_facet_norm","u_inf_facet_norm","grad_u_L2_norm","grad_p_L2_norm");
  printf("%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n\n",
      Stats.hmean,  Stats.u_L2_norm,  Stats.p_L2_norm,  Stats.u_inf_norm,  Stats.p_inf_norm,  Stats.u_L2_facet_norm,  Stats.u_inf_facet_norm,  Stats.grad_u_L2_norm,  Stats.grad_p_L2_norm);

  ////if (unsteady)
  //{
  //  cout << "\nmean errors: \n";
  //  cout << "# hmean            u_L2_err         p_L2_err          grad_u_L2_err     grad_p_L2_err" << endl;
  //  printf( "%.15lf %.15lf %.15lf %.15lf %.15lf\n", Stats.mean_hmean() , Stats.mean_u_L2_norm(), Stats.mean_p_L2_norm(), Stats.mean_grad_u_L2_norm(), Stats.mean_grad_p_L2_norm());
  //}


  PetscFunctionReturn(0);
}

void AppCtx::computeError(Vec const& Vec_x, Vec &Vec_up, double tt)
{
  MatrixXd            u_coefs_c(n_dofs_u_per_cell/dim, dim);
  MatrixXd            u_coefs_c_trans(dim,n_dofs_u_per_cell/dim);
  MatrixXd            u_coefs_f(n_dofs_u_per_facet/dim, dim);
  MatrixXd            u_coefs_f_trans(dim,n_dofs_u_per_facet/dim);
  VectorXd            p_coefs_c(n_dofs_p_per_cell);
  MatrixXd            x_coefs_c(nodes_per_cell, dim);
  MatrixXd            x_coefs_c_trans(dim, nodes_per_cell);
  MatrixXd            x_coefs_f(nodes_per_facet, dim);
  MatrixXd            x_coefs_f_trans(dim, nodes_per_facet);
  MatrixXd            ftau_coefs_f(nodes_per_facet, dim);
  MatrixXd            ftau_coefs_f_trans(dim, nodes_per_facet);
  Tensor              F_c(dim,dim), invF_c(dim,dim), invFT_c(dim,dim);
  Tensor              F_f(dim,dim-1), invF_f(dim-1,dim), fff(dim-1,dim-1);
  MatrixXd            dxphi_err(n_dofs_u_per_cell/dim, dim);
  MatrixXd            dxpsi_err(n_dofs_p_per_cell, dim);
  MatrixXd            dxqsi_err(nodes_per_cell, dim);
  Tensor              dxU(dim,dim); // grad u
  Vector              dxP(dim);     // grad p
  Vector              Xqp(dim);
  Vector              Uqp(dim);
  Vector              Ftauqp(dim);

  double              Pqp;
  VectorXi            cell_nodes(nodes_per_cell);
  double              J, JxW;
  double              weight;
  int                 tag;
  double              volume=0;

  double              p_L2_norm = 0.;
  double              u_L2_norm = 0.;
  double              grad_u_L2_norm = 0.;
  double              grad_p_L2_norm = 0.;
  double              p_inf_norm = 0.;
  double              u_inf_norm = 0.;
  double              hmean = 0.;
  double              u_L2_facet_norm = 0.;
  double              u_inf_facet_norm = 0.;
  double              u_exact_L2_norm = 0.;
  double              u_exact_inf_norm = 0.;
  double              ftau_L2_facet_norm = 0.;
  double              ftau_inf_facet_norm = 0.;

  VectorXi            mapU_c(n_dofs_u_per_cell);
  VectorXi            mapU_f(n_dofs_u_per_facet);
  VectorXi            mapU_r(n_dofs_u_per_corner);
  VectorXi            mapP_c(n_dofs_p_per_cell);
  VectorXi            mapP_r(n_dofs_p_per_corner);
  VectorXi            mapM_c(dim*nodes_per_cell);
  VectorXi            mapM_f(dim*nodes_per_facet);
  VectorXi            mapM_r(dim*nodes_per_corner);

  int                 tag_pt0, tag_pt1, tag_pt2, nPer = 0, ccell, nod_id;
  double const*       Xqpb;  //coordonates at the master element \hat{X}
  Vector              Phi(dim), DPhi(dim), X0(dim), X2(dim), Xcc(3), Vdat(3), normal(dim), tangent(dim);
  Tensor              F_c_curv(dim,dim);
  bool                curvf = false;
  //Permutation matrices
  TensorXi            PerM3(TensorXi::Zero(3,3)), PerM6(TensorXi::Zero(6,6));
  MatrixXi            PerM12(MatrixXi::Zero(12,12));
  PerM3(0,1) = 1; PerM3(1,2) = 1; PerM3(2,0) = 1;
  PerM6(0,2) = 1; PerM6(1,3) = 1; PerM6(2,4) = 1; PerM6(3,5) = 1; PerM6(4,0) = 1; PerM6(5,1) = 1;
  PerM12.block(0,0,6,6) = PerM6; PerM12.block(6,6,6,6) = PerM6;
  int                 is_slipvel, is_fsi;
  Tensor              F_f_curv(dim,dim-1);
  double              ybar = 0.0;


  VecSetOption(Vec_up, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE); //?

  ////////////////////////////////////////////////// STARTING CELL ITERATION //////////////////////////////////////////////////
  cell_iterator cell = mesh->cellBegin();
  cell_iterator cell_end = mesh->cellEnd();
  for (; cell != cell_end; ++cell)
  {
    tag = cell->getTag();

    if (is_sfip)
      if (is_in(tag,solidonly_tags))
        continue;

    //get nodal coordinates of the old and (permuted) new cell//////////////////////////////////////////////////
    mesh->getCellNodesId(&*cell, cell_nodes.data());  //cout << cell_nodes.transpose() << endl;
    if (is_curvt){
      //find good orientation for nodes in case of curved border element
      tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();
      tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
      tag_pt2 = mesh->getNodePtr(cell_nodes(2))->getTag();
      //test if the cell is acceptable (one curved side)
      //bcell = is_in(tag_pt0,fluidonly_tags)+is_in(tag_pt1,fluidonly_tags)+is_in(tag_pt2,fluidonly_tags);
      ccell = is_in(tag_pt0,flusoli_tags)+is_in(tag_pt1,flusoli_tags)+is_in(tag_pt2,flusoli_tags)
             +is_in(tag_pt0,slipvel_tags)+is_in(tag_pt1,slipvel_tags)+is_in(tag_pt2,slipvel_tags)
             +is_in(tag_pt0,interface_tags)+is_in(tag_pt1,interface_tags)+is_in(tag_pt2,interface_tags);
      curvf = /*bcell==1 &&*/ ccell==2;  nPer = 0;
      if (curvf){
        while ( !(is_in(tag_pt1, fluidonly_tags) || is_in(tag_pt1, feature_tags)
               || is_in(tag_pt1, dirichlet_tags) || is_in(tag_pt1, neumann_tags)) ){
          cell_nodes = PerM3*cell_nodes; nPer++;  //counts how many permutations are performed
          tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
        }
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //mapeamento do local para o global//////////////////////////////////////////////////
    dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);
    dof_handler[DH_UNKM].getVariable(VAR_U).getCellDofs(mapU_c.data(), &*cell);
    dof_handler[DH_UNKM].getVariable(VAR_P).getCellDofs(mapP_c.data(), &*cell);

    //if cell is permuted, this corrects the maps; mapZ is alreday corrected by hand in the previous if conditional
    if (curvf){
      for (int l = 0; l < nPer; l++){
        mapM_c = PerM6*mapM_c;  //cout << mapM_c.transpose() << endl;
        mapU_c = PerM6*mapU_c;  //cout << mapU_c.transpose() << endl;
        mapP_c = PerM3*mapP_c;  //cout << mapP_c.transpose() << endl;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //get the value for the variables//////////////////////////////////////////////////
    VecGetValues(Vec_up, mapU_c.size(), mapU_c.data(), u_coefs_c.data());
    VecGetValues(Vec_up, mapP_c.size(), mapP_c.data(), p_coefs_c.data());
    VecGetValues(Vec_x,  mapM_c.size(), mapM_c.data(), x_coefs_c.data());

    u_coefs_c_trans = u_coefs_c.transpose();
    x_coefs_c_trans = x_coefs_c.transpose();

    if (curvf){
      tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();
      nod_id = is_in_id(tag_pt0,flusoli_tags)+is_in_id(tag_pt0,slipvel_tags);
      Xcc = XG_0[nod_id-1];
      X0(0) = x_coefs_c_trans(0,0);  X0(1) = x_coefs_c_trans(1,0);
      X2(0) = x_coefs_c_trans(0,2);  X2(1) = x_coefs_c_trans(1,2);
      Vdat << RV[nod_id-1](0),RV[nod_id-1](1), 0.0; //theta_0[nod_id-1]; //container for R1, R2, theta
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
    for (int qp = 0; qp < n_qpts_err; ++qp)
    {
      F_c = Tensor::Zero(dim,dim);  //Zero(dim,dim);
      Xqp = Vector::Zero(dim);// coordenada espacial (x,y,z) do ponto de quadratura
      if (curvf){//F_c_curv.setZero();
        Xqpb = quadr_err->point(qp);
        Phi = curved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
        DPhi = Dcurved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
        F_c_curv.col(0) = -Phi;
        F_c_curv.col(1) = -Phi + (1.0-Xqpb[0]-Xqpb[1])*DPhi;
        F_c = F_c_curv;
        Xqp = (1.0-Xqpb[0]-Xqpb[1])*Phi;
      }

      F_c   += x_coefs_c_trans * dLqsi_err[qp];
      J      = F_c.determinant();
      invF_c = F_c.inverse();
      invFT_c= invF_c.transpose();

      dxphi_err = dLphi_err[qp] * invF_c;
      dxpsi_err = dLpsi_err[qp] * invF_c;
      dxqsi_err = dLqsi_err[qp] * invF_c;

      dxU  = u_coefs_c_trans * dxphi_err;       // n+utheta
      dxP  = dxpsi_err.transpose() * p_coefs_c;

      Xqp += x_coefs_c_trans * qsi_err[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
      Uqp  = u_coefs_c_trans * phi_err[qp];
      Pqp  = p_coefs_c.dot(psi_err[qp]);

      //quadrature weight//////////////////////////////////////////////////
      weight = quadr_err->weight(qp);
      JxW = J*weight;
      if (is_axis){
        JxW = JxW*2.0*pi*Xqp(0);
      }
      //////////////////////////////////////////////////

      //NORMS//////////////////////////////////////////////////
      //cout << Xqp.transpose() << "   " << u_exacta(Xqp, tt, tag).transpose() << "   " << Uqp.transpose() << endl;
      //  note: norm(u, H1)^2 = norm(u, L2)^2 + norm(gradLphi_c, L2)^2
      u_L2_norm        += (u_exacta(Xqp, tt, tag) - Uqp).squaredNorm()*JxW;      //cambiar a u_exact
      p_L2_norm        += sqr(p_exacta(Xqp, tt, tag) - Pqp)*JxW;
      grad_u_L2_norm   += (grad_u_exacta(Xqp, tt, tag) - dxU).squaredNorm()*JxW;
      grad_p_L2_norm   += (grad_p_exact(Xqp, tt, tag) - dxP).squaredNorm()*JxW;
      u_inf_norm        = max(u_inf_norm, (u_exacta(Xqp, tt, tag) - Uqp).norm()); //cambiar a u_exact
      p_inf_norm        = max(p_inf_norm, fabs(p_exact(Xqp, tt, tag) - Pqp));
      u_exact_L2_norm  += (u_exacta(Xqp, tt, tag)).squaredNorm()*JxW;
      u_exact_inf_norm  = max(u_exact_inf_norm,(u_exacta(Xqp, tt, tag)).norm());

      volume += JxW;
    } // fim quadratura //////////////////////////////////////////////////
  } // end elementos //////////////////////////////////////////////////
  ////////////////////////////////////////////////// ENDING CELL ITERATION //////////////////////////////////////////////////

  ////////////////////////////////////////////////// STARTING FACET ITERATION //////////////////////////////////////////////////
  facet_iterator facet = mesh->facetBegin();
  facet_iterator facet_end = mesh->facetEnd();
  for (; facet != facet_end; ++facet)
  {
    tag = facet->getTag();

    if ( !(is_in(tag, dirichlet_tags) || is_in(tag, neumann_tags) || is_in(tag, interface_tags) ||
          is_in(tag, solid_tags) || is_in(tag, periodic_tags) || is_in(tag, feature_tags) ||
          is_in(tag, flusoli_tags) || is_in(tag, slipvel_tags)) )
      continue;

    is_slipvel = is_in_id(tag, slipvel_tags);
    is_fsi     = is_in_id(tag, flusoli_tags);

    // mapeamento do local para o global//////////////////////////////////////////////////
    dof_handler[DH_UNKM].getVariable(VAR_U).getFacetDofs(mapU_f.data(), &*facet);
    dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(mapM_f.data(), &*facet);

    if (is_curvt){
      if (is_fsi || is_slipvel){curvf = true;}
      else {curvf = false;}
    }
    //////////////////////////////////////////////////

    //get the value for the variables//////////////////////////////////////////////////
    VecGetValues(Vec_up, mapU_f.size(), mapU_f.data(), u_coefs_f.data());
    VecGetValues(Vec_x,  mapM_f.size(), mapM_f.data(), x_coefs_f.data());
    VecGetValues(Vec_ftau_0,  mapM_f.size(), mapM_f.data(), ftau_coefs_f.data());

    u_coefs_f_trans = u_coefs_f.transpose();
    x_coefs_f_trans = x_coefs_f.transpose();
    ftau_coefs_f_trans = ftau_coefs_f.transpose();

    if (curvf){
      Xcc = XG_0[is_fsi+is_slipvel-1];
      X0(0) = x_coefs_f_trans(0,0);  X0(1) = x_coefs_f_trans(1,0);
      X2(0) = x_coefs_f_trans(0,1);  X2(1) = x_coefs_f_trans(1,1);  //cout << Xcc.transpose() << "   " << X0.transpose() << " " << X2.transpose() << endl;
      Vdat << RV[is_fsi+is_slipvel-1](0),RV[is_fsi+is_slipvel-1](1), 0.0; //theta_0[nod_id-1]; //container for R1, R2, theta
    }
    //////////////////////////////////////////////////

    ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
    for (int qp = 0; qp < n_qpts_facet; ++qp)
    {
      F_f = Tensor::Zero(dim,dim-1);  //Zero(dim,dim);
      Xqp = Vector::Zero(dim);// coordenada espacial (x,y,z) do ponto de quadratura
      if (curvf){//F_c_curv.setZero();
        Xqpb = quadr_facet->point(qp);  //cout << Xqpb[0] << " " << Xqpb[1] << endl;
        ybar = (1.0+Xqpb[0])/2.0;
        Phi = curved_Phi(ybar,X0,X2,Xcc,Vdat,dim);
        DPhi = Dcurved_Phi(ybar,X0,X2,Xcc,Vdat,dim);
        F_f_curv.col(0) = -Phi/2.0 + (1.0-ybar)*DPhi/2.0;
        F_f = F_f_curv;
        Xqp = (1.0-ybar)*Phi;
      }

      F_f += x_coefs_f_trans * dLqsi_f[qp];
      fff  = F_f.transpose()*F_f;
      J    = sqrt(fff.determinant());

      Xqp += x_coefs_f_trans * qsi_f[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
      Uqp  = u_coefs_f_trans * phi_f[qp];

      //quadrature weight//////////////////////////////////////////////////
      weight = quadr_facet->weight(qp);
      JxW = J*weight;
      if (is_axis){
        JxW = JxW*2.0*pi*Xqp(0);
      }
      //////////////////////////////////////////////////

      Vector U_exact = u_exacta(Xqp, tt, tag);
      u_L2_facet_norm += (U_exact - Uqp).squaredNorm()*JxW;
      //u_L2_facet_norm += JxW;
      double const diff = (U_exact - Uqp).norm();
      if (diff > u_inf_facet_norm)
        u_inf_facet_norm = diff;

      if (is_fsi || is_slipvel){
        Ftauqp = ftau_coefs_f_trans * phi_f[qp];
        normal(0) = +F_f(1,0);
        normal(1) = -F_f(0,0);
        normal.normalize();  //cout << normal.transpose() << endl;
        normal = -normal;  //... now this normal points OUT the body
        //tangent(0) = -normal(1); tangent(1) = normal(0);

        Vector ftau_exact = ftau_exacta(Xqp, XG_1[is_fsi+is_slipvel-1], normal,
                                        tt, tag, theta_1[is_fsi+is_slipvel-1]);
        ftau_L2_facet_norm = +(ftau_exact - Ftauqp).squaredNorm()*JxW;
        double const difftau = (ftau_exact - Ftauqp).norm();
        if (difftau > ftau_inf_facet_norm)
          ftau_inf_facet_norm = difftau;

      }


    } // fim quadratura
  }
  ////////////////////////////////////////////////// ENDING FACET ITERATION //////////////////////////////////////////////////

  u_L2_norm      = sqrt(u_L2_norm     );
  p_L2_norm      = sqrt(p_L2_norm     );
  grad_u_L2_norm = sqrt(grad_u_L2_norm);
  grad_p_L2_norm = sqrt(grad_p_L2_norm);
  u_L2_facet_norm = sqrt(u_L2_facet_norm);
  ftau_L2_facet_norm = sqrt(ftau_L2_facet_norm);

  // ASSUME QUE SÓ POSSA TER NO MÁXIMO 1 NÓ POR ARESTA
  VectorXi edge_nodes(3);
  Vector Xa(dim), Xb(dim);
  int n_edges=0;

  if (dim==2)
  //FEP_PRAGMA_OMP(parallel default(none) shared(hmean))
  {
    const int n_edges_total = mesh->numFacetsTotal();
    Facet const* edge(NULL);

    //FEP_PRAGMA_OMP(for nowait)
    for (int a = 0; a < n_edges_total; ++a)
    {
      edge = mesh->getFacetPtr(a);
      if (edge->isDisabled())
        continue;

      mesh->getFacetNodesId(&*edge, edge_nodes.data());

      mesh->getNodePtr(edge_nodes[0])->getCoord(Xa.data(),dim);
      mesh->getNodePtr(edge_nodes[1])->getCoord(Xb.data(),dim);
      hmean += (Xa-Xb).norm();
      ++n_edges;
    }
  }
  else
  if (dim==3)
  //FEP_PRAGMA_OMP(parallel default(none) shared(cout,hmean))
  {
    const int n_edges_total = mesh->numCornersTotal();
    Corner const* edge(NULL);
//cout << "aqui" << endl;
    //FEP_PRAGMA_OMP(for nowait)
    for (int a = 0; a < n_edges_total; ++a)
    {
      edge = mesh->getCornerPtr(a);
      if (edge->isDisabled())
        continue;

      mesh->getCornerNodesId(&*edge, edge_nodes.data());

      mesh->getNodePtr(edge_nodes[0])->getCoord(Xa.data(),dim);
      mesh->getNodePtr(edge_nodes[1])->getCoord(Xb.data(),dim);
      hmean += (Xa-Xb).norm();
      ++n_edges;
    }
  }
  hmean /= n_edges;
  //hme = hmean;
  //if (time_step==maxts)
  {
    cout << endl;
    printf("%-21s %-21s %-21s %-21s %-21s %-21s %-21s %-21s %-21s %-21s %-21s\n",
              "# hmean","u_L2_norm","p_L2_norm","u_inf_norm","p_inf_norm",
              "u_L2_facet_norm","u_inf_facet_norm","grad_u_L2_norm","grad_p_L2_norm",
              "ftau_L2_facet_norm","ftau_inf_facet_norm");
    printf("%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n\n",
                 hmean,  u_L2_norm,  p_L2_norm,  u_inf_norm,  p_inf_norm,
                 u_L2_facet_norm,  u_inf_facet_norm,  grad_u_L2_norm,  grad_p_L2_norm,
                 ftau_L2_facet_norm,  ftau_inf_facet_norm);
    //printf("%-21s %-21s %-21s \n", "# hmean", "u_L2_norm", "u_inf_facet_norm", "p_L2_norm", );
    //printf("%.15e %.15e %.15e \n\n", hmean, u_L2_norm, u_inf_norm);
  }

  Stats.add_p_L2_norm        (p_L2_norm       );
  Stats.add_u_L2_norm        (u_L2_norm       );
  Stats.add_grad_u_L2_norm   (grad_u_L2_norm  );
  Stats.add_grad_p_L2_norm   (grad_p_L2_norm  );
  Stats.add_p_inf_norm       (p_inf_norm      );
  Stats.add_u_inf_norm       (u_inf_norm      );
  Stats.add_hmean            (hmean           );
  Stats.add_u_L2_facet_norm  (u_L2_facet_norm );
  Stats.add_u_inf_facet_norm (u_inf_facet_norm);

  filr.open(errc,iostream::app);
  filr.precision(15);
  filr.setf(iostream::fixed, iostream::floatfield);
  filr << pow(2.0,time_step) << " " << u_L2_norm << " " << p_L2_norm << " " << u_inf_norm << " " << p_inf_norm
       << " " << u_L2_facet_norm << " " << u_inf_facet_norm << " " << u_exact_L2_norm << " " << u_exact_inf_norm
       << " " << ftau_L2_facet_norm << " " << ftau_inf_facet_norm;
  filr << endl;
  filr.close();

}

void AppCtx::computeForces(Vec const& Vec_x, Vec &Vec_up)
{
  int                 tag;
  VectorXi            mapU_f(n_dofs_u_per_facet);
  VectorXi            mapP_f(n_dofs_p_per_facet);
  VectorXi            mapM_f(dim*nodes_per_facet);
  MatrixXd            u_coefs_f(n_dofs_u_per_facet/dim, dim);
  VectorXd            p_coefs_f(n_dofs_p_per_facet);
  MatrixXd            x_coefs_f(nodes_per_facet, dim);
  MatrixXd            ftau_coefs_f(nodes_per_facet, dim);
  MatrixXd            sv_coefs_f(nodes_per_facet, dim);
  MatrixXd            u_coefs_f_trans(dim,n_dofs_u_per_facet/dim);
  MatrixXd            x_coefs_f_trans(dim, nodes_per_facet);
  MatrixXd            ftau_coefs_f_trans(dim, nodes_per_facet);
  MatrixXd            sv_coefs_f_trans(dim, nodes_per_facet);
  MatrixXd            dxphi_f(n_dofs_u_per_facet/dim, dim);
  Tensor              F_f(dim,dim-1), invF_f(dim-1,dim), fff(dim-1,dim-1), dxU_f(dim,dim);
  Vector              normal(dim), Traction_(Vector::Zero(dim)), Xqp(dim), ftauqp(dim), svqp(dim);// Uqp(dim);
  double              J, JxW, weight, areas = 0.0;
  //Tensor              I(Tensor::Identity(dim,dim));
  double              VD = 0.0;  //viscous dissipation

  facet_iterator facet = mesh->facetBegin();
  facet_iterator facet_end = mesh->facetEnd();
  for (; facet != facet_end; ++facet)
  {
    tag = facet->getTag();

    if (!(is_in(tag, flusoli_tags) || is_in(tag, slipvel_tags)))
      continue;

    dof_handler[DH_UNKM].getVariable(VAR_U).getFacetDofs(mapU_f.data(), &*facet);
    dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(mapM_f.data(), &*facet);
    dof_handler[DH_UNKM].getVariable(VAR_P).getFacetDofs(mapP_f.data(), &*facet);

    VecGetValues(Vec_up, mapU_f.size(), mapU_f.data(), u_coefs_f.data());
    VecGetValues(Vec_up, mapP_f.size(), mapP_f.data(), p_coefs_f.data());
    VecGetValues(Vec_x,  mapM_f.size(), mapM_f.data(), x_coefs_f.data());
    VecGetValues(Vec_ftau_0,  mapM_f.size(), mapM_f.data(), ftau_coefs_f.data());
    VecGetValues(Vec_slipv_0,  mapM_f.size(), mapM_f.data(), sv_coefs_f.data());

    u_coefs_f_trans = u_coefs_f.transpose();
    x_coefs_f_trans = x_coefs_f.transpose();
    ftau_coefs_f_trans = ftau_coefs_f.transpose();
    sv_coefs_f_trans = sv_coefs_f.transpose();

    for (int qp = 0; qp < n_qpts_facet; ++qp)
    {
      F_f  = x_coefs_f_trans * dLqsi_f[qp];
      fff  = F_f.transpose()*F_f;
      J    = sqrt(fff.determinant());
      invF_f = fff.inverse()*F_f.transpose();

      if (dim==2)
      {
        normal(0) = +F_f(1,0);
        normal(1) = -F_f(0,0);
        normal.normalize();
        normal = -normal;  //... now this normal points OUT the body
      }
      else
      {
        normal = cross(F_f.col(0), F_f.col(1));
        normal.normalize();
      }

      Xqp     = x_coefs_f_trans * qsi_f[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
      //Uqp  = u_coefs_f_trans * phi_f[qp];
      ftauqp  = ftau_coefs_f_trans * phi_f[qp];
      svqp    = sv_coefs_f_trans * phi_f[qp];

      weight = quadr_facet->weight(qp);
      JxW = J*weight;
      if (is_axis){
        JxW = JxW*2.0*pi*Xqp(0);
      }

      dxphi_f = dLphi_f[qp] * invF_f;
      dxU_f   = u_coefs_f_trans * dxphi_f; // n+utheta
      areas += JxW;
      Traction_ += JxW * ( -p_coefs_f.dot(psi_f[qp])*normal + muu(tag)*(dxU_f + dxU_f.transpose())*normal );
      VD += JxW*(ftauqp.dot(svqp));
      //cout << Traction_.transpose() << "  " << areas << endl;
    }
  }
  //cout << "Traction Force = " << Traction_.transpose() << ". Area = " << areas
  //     << ". Viscous Dissipation from ftau*us = " << VD << endl;
  //filf.open(forc,iostream::app);
  //filf.precision(15);
  //filf << current_time << " " << Traction_.transpose() << " " << VD << " ";
  //filf.close();
}

void AppCtx::computeViscousDissipation(Vec const& Vec_x, Vec &Vec_up)
{
  MatrixXd            u_coefs_c(n_dofs_u_per_cell/dim, dim);
  MatrixXd            u_coefs_c_trans(dim,n_dofs_u_per_cell/dim);
  //MatrixXd            u_coefs_f(n_dofs_u_per_facet/dim, dim);
  //MatrixXd            u_coefs_f_trans(dim,n_dofs_u_per_facet/dim);
  VectorXd            p_coefs_c(n_dofs_p_per_cell);
  MatrixXd            x_coefs_c(nodes_per_cell, dim);
  MatrixXd            x_coefs_c_trans(dim, nodes_per_cell);
  //MatrixXd            x_coefs_f(nodes_per_facet, dim);
  //MatrixXd            x_coefs_f_trans(dim, nodes_per_facet);
  Tensor              F_c(dim,dim), invF_c(dim,dim), invFT_c(dim,dim);
  //Tensor              F_f(dim,dim-1), invF_f(dim-1,dim), fff(dim-1,dim-1);
  MatrixXd            dxphi_err(n_dofs_u_per_cell/dim, dim);
  MatrixXd            dxpsi_err(n_dofs_p_per_cell, dim);
  //MatrixXd            dxqsi_err(nodes_per_cell, dim);
  Tensor              dxU(dim,dim), dxUs(dim,dim); // grad u
  Vector              dxP(dim);     // grad p
  Vector              Xqp(dim);
  Vector              Uqp(dim);
  double              Pqp;
  VectorXi            cell_nodes(nodes_per_cell);
  double              J, JxW;
  double              weight;
  int                 tag;

  VectorXi            mapU_c(n_dofs_u_per_cell);
  //VectorXi            mapU_f(n_dofs_u_per_facet);
  //VectorXi            mapU_r(n_dofs_u_per_corner);
  VectorXi            mapP_c(n_dofs_p_per_cell);
  //VectorXi            mapP_r(n_dofs_p_per_corner);
  VectorXi            mapM_c(dim*nodes_per_cell);
  //VectorXi            mapM_f(dim*nodes_per_facet);
  //VectorXi            mapM_r(dim*nodes_per_corner);

  int                 tag_pt0, tag_pt1, tag_pt2, nPer = 0, ccell, nod_id;
  double const*       Xqpb;  //coordonates at the master element \hat{X}
  Vector              Phi(dim), DPhi(dim), X0(dim), X2(dim), Xcc(3), Vdat(3);
  Tensor              F_c_curv(dim,dim);
  bool                curvf = false;
  //Permutation matrices
  TensorXi            PerM3(TensorXi::Zero(3,3)), PerM6(TensorXi::Zero(6,6));
  MatrixXi            PerM12(MatrixXi::Zero(12,12));
  PerM3(0,1) = 1; PerM3(1,2) = 1; PerM3(2,0) = 1;
  PerM6(0,2) = 1; PerM6(1,3) = 1; PerM6(2,4) = 1; PerM6(3,5) = 1; PerM6(4,0) = 1; PerM6(5,1) = 1;
  PerM12.block(0,0,6,6) = PerM6; PerM12.block(6,6,6,6) = PerM6;
  Tensor              F_f_curv(dim,dim-1);

  double    VD = 0.0, visc = 0.0, ND = 0.0, VT = 0.0;  //viscous dissipation

  VecSetOption(Vec_up, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE); //?

  ////////////////////////////////////////////////// STARTING CELL ITERATION //////////////////////////////////////////////////
  cell_iterator cell = mesh->cellBegin();
  cell_iterator cell_end = mesh->cellEnd();
  for (; cell != cell_end; ++cell)
  {
    tag = cell->getTag();

    if (is_sfip)
      if (is_in(tag,solidonly_tags))
        continue;

    //get nodal coordinates of the old and (permuted) new cell//////////////////////////////////////////////////
    mesh->getCellNodesId(&*cell, cell_nodes.data());  //cout << cell_nodes.transpose() << endl;
    if (is_curvt){
      //find good orientation for nodes in case of curved border element
      tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();
      tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
      tag_pt2 = mesh->getNodePtr(cell_nodes(2))->getTag();
      //test if the cell is acceptable (one curved side)
      //bcell = is_in(tag_pt0,fluidonly_tags)+is_in(tag_pt1,fluidonly_tags)+is_in(tag_pt2,fluidonly_tags);
      ccell = is_in(tag_pt0,flusoli_tags)+is_in(tag_pt1,flusoli_tags)+is_in(tag_pt2,flusoli_tags)
             +is_in(tag_pt0,slipvel_tags)+is_in(tag_pt1,slipvel_tags)+is_in(tag_pt2,slipvel_tags)
             +is_in(tag_pt0,interface_tags)+is_in(tag_pt1,interface_tags)+is_in(tag_pt2,interface_tags);
      curvf = /*bcell==1 &&*/ ccell==2;  nPer = 0;
      if (curvf){
        while ( !(is_in(tag_pt1, fluidonly_tags) || is_in(tag_pt1, feature_tags)
               || is_in(tag_pt1, dirichlet_tags) || is_in(tag_pt1, neumann_tags)) ){
          cell_nodes = PerM3*cell_nodes; nPer++;  //counts how many permutations are performed
          tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
        }
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //mapeamento do local para o global//////////////////////////////////////////////////
    dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);
    dof_handler[DH_UNKM].getVariable(VAR_U).getCellDofs(mapU_c.data(), &*cell);
    dof_handler[DH_UNKM].getVariable(VAR_P).getCellDofs(mapP_c.data(), &*cell);

    //if cell is permuted, this corrects the maps; mapZ is alreday corrected by hand in the previous if conditional
    if (curvf){
      for (int l = 0; l < nPer; l++){
        mapM_c = PerM6*mapM_c;  //cout << mapM_c.transpose() << endl;
        mapU_c = PerM6*mapU_c;  //cout << mapU_c.transpose() << endl;
        mapP_c = PerM3*mapP_c;  //cout << mapP_c.transpose() << endl;
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //get the value for the variables//////////////////////////////////////////////////
    VecGetValues(Vec_up, mapU_c.size(), mapU_c.data(), u_coefs_c.data());
    VecGetValues(Vec_up, mapP_c.size(), mapP_c.data(), p_coefs_c.data());
    VecGetValues(Vec_x,  mapM_c.size(), mapM_c.data(), x_coefs_c.data());

    u_coefs_c_trans = u_coefs_c.transpose();
    x_coefs_c_trans = x_coefs_c.transpose();

    if (curvf){
      tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();
      nod_id = is_in_id(tag_pt0,flusoli_tags)+is_in_id(tag_pt0,slipvel_tags);
      Xcc = XG_0[nod_id-1];
      X0(0) = x_coefs_c_trans(0,0);  X0(1) = x_coefs_c_trans(1,0);
      X2(0) = x_coefs_c_trans(0,2);  X2(1) = x_coefs_c_trans(1,2);
      Vdat << RV[nod_id-1](0),RV[nod_id-1](1), 0.0; //theta_0[nod_id-1]; //container for R1, R2, theta
    }

    visc = muu(tag);

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
    for (int qp = 0; qp < n_qpts_cell; ++qp)
    {
      F_c = Tensor::Zero(dim,dim);  //Zero(dim,dim);
      Xqp = Vector::Zero(dim);// coordenada espacial (x,y,z) do ponto de quadratura
      if (curvf){//F_c_curv.setZero();
       Xqpb = quadr_err->point(qp);
       Phi = curved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
       DPhi = Dcurved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
       F_c_curv.col(0) = -Phi;
       F_c_curv.col(1) = -Phi + (1.0-Xqpb[0]-Xqpb[1])*DPhi;
       F_c = F_c_curv;
       Xqp = (1.0-Xqpb[0]-Xqpb[1])*Phi;
      }

      F_c   += x_coefs_c_trans * dLqsi_c[qp];
      J      = F_c.determinant();
      invF_c = F_c.inverse();
      invFT_c= invF_c.transpose();

      dxphi_err = dLphi_c[qp] * invF_c;
      dxpsi_err = dLpsi_c[qp] * invF_c;
      //dxqsi_err = dLqsi_err[qp] * invF_c;

      dxU  = u_coefs_c_trans * dxphi_err;       // n+utheta
      dxP  = dxpsi_err.transpose() * p_coefs_c;

      Xqp += x_coefs_c_trans * qsi_c[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
      Uqp  = u_coefs_c_trans * phi_c[qp];
      Pqp  = p_coefs_c.dot(psi_c[qp]);

      //quadrature weight//////////////////////////////////////////////////
      weight = quadr_cell->weight(qp);
      JxW = J*weight;
      if (is_axis){
       JxW = JxW*2.0*pi*Xqp(0);
      }
      //////////////////////////////////////////////////
      //cout << DobCont((dxU.transpose()+dxU)/2.0,dxU) << endl;
      double pdiv = Pqp*dxU.trace();
      dxUs = (dxU + dxU.transpose())/2.0;

      //cout << DobCont(dxU,dxU) << endl;
      ND += JxW*pdiv; //numeric dissipation
      VD += JxW*2.0*visc*DobCont(dxUs,dxUs);
      VT += JxW*( 2.0*visc*DobCont(dxUs,dxUs) - pdiv);  //cout << pdiv << endl;
      if (is_axis){
        VT += JxW*( 2.0*visc*Uqp(0)*Uqp(0)/(Xqp(0)*Xqp(0)) - Pqp*Uqp(0)/Xqp(0) ); //cout << Pqp*Uqp(0)/Xqp(0) << endl;
      }

     } // fim quadratura //////////////////////////////////////////////////
   } // end elementos //////////////////////////////////////////////////


  // Power //////////////////////////////////////////////////////////////////////////////////////////////
  //int                 tag;
  VectorXi            mapU_f(n_dofs_u_per_facet);
  VectorXi            mapP_f(n_dofs_p_per_facet);
  VectorXi            mapM_f(dim*nodes_per_facet);
  MatrixXd            u_coefs_f(n_dofs_u_per_facet/dim, dim);
  VectorXd            p_coefs_f(n_dofs_p_per_facet);
  MatrixXd            x_coefs_f(nodes_per_facet, dim);
  MatrixXd            ftau_coefs_f(nodes_per_facet, dim);
  MatrixXd            sv_coefs_f(nodes_per_facet, dim);
  MatrixXd            u_coefs_f_trans(dim,n_dofs_u_per_facet/dim);
  MatrixXd            x_coefs_f_trans(dim, nodes_per_facet);
  MatrixXd            ftau_coefs_f_trans(dim, nodes_per_facet);
  MatrixXd            sv_coefs_f_trans(dim, nodes_per_facet);
  MatrixXd            dxphi_f(n_dofs_u_per_facet/dim, dim);
  Tensor              F_f(dim,dim-1), invF_f(dim-1,dim), fff(dim-1,dim-1), dxU_f(dim,dim);
  Vector              normal(dim), Traction_(Vector::Zero(dim))/*, Xqp(dim)*/, ftauqp(dim), svqp(dim);// Uqp(dim);
  double              /*J, JxW, weight, */areas = 0.0;
  //Tensor              I(Tensor::Identity(dim,dim));
  double              PD = 0.0, PDu = 0.0;  //viscous dissipation

  facet_iterator facet = mesh->facetBegin();
  facet_iterator facet_end = mesh->facetEnd();
  for (; facet != facet_end; ++facet)
  {
    tag = facet->getTag();

    if (!(is_in(tag, flusoli_tags) || is_in(tag, slipvel_tags)))
      continue;

    dof_handler[DH_UNKM].getVariable(VAR_U).getFacetDofs(mapU_f.data(), &*facet);
    dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(mapM_f.data(), &*facet);
    dof_handler[DH_UNKM].getVariable(VAR_P).getFacetDofs(mapP_f.data(), &*facet);

    VecGetValues(Vec_up, mapU_f.size(), mapU_f.data(), u_coefs_f.data());
    VecGetValues(Vec_up, mapP_f.size(), mapP_f.data(), p_coefs_f.data());
    VecGetValues(Vec_x,  mapM_f.size(), mapM_f.data(), x_coefs_f.data());
    VecGetValues(Vec_ftau_0,  mapM_f.size(), mapM_f.data(), ftau_coefs_f.data());
    VecGetValues(Vec_slipv_0,  mapM_f.size(), mapM_f.data(), sv_coefs_f.data());

    u_coefs_f_trans = u_coefs_f.transpose();
    x_coefs_f_trans = x_coefs_f.transpose();
    ftau_coefs_f_trans = ftau_coefs_f.transpose();
    sv_coefs_f_trans = sv_coefs_f.transpose();

    for (int qp = 0; qp < n_qpts_facet; ++qp)
    {
      F_f  = x_coefs_f_trans * dLqsi_f[qp];
      fff  = F_f.transpose()*F_f;
      J    = sqrt(fff.determinant());
      invF_f = fff.inverse()*F_f.transpose();

      if (dim==2)
      {
        normal(0) = +F_f(1,0);
        normal(1) = -F_f(0,0);
        normal.normalize();
        normal = -normal;  //... now this normal points OUT the body
      }
      else
      {
        normal = cross(F_f.col(0), F_f.col(1));
        normal.normalize();
      }

      Xqp     = x_coefs_f_trans * qsi_f[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
      Uqp     = u_coefs_f_trans * phi_f[qp];
      ftauqp  = ftau_coefs_f_trans * phi_f[qp];
      svqp    = sv_coefs_f_trans * phi_f[qp];

      weight = quadr_facet->weight(qp);
      JxW = J*weight;
      if (is_axis){
        JxW = JxW*2.0*pi*Xqp(0);
      }

      dxphi_f = dLphi_f[qp] * invF_f;
      dxU_f   = u_coefs_f_trans * dxphi_f; // n+utheta
      areas += JxW;
      Traction_ += JxW * ( -p_coefs_f.dot(psi_f[qp])*normal + muu(tag)*(dxU_f + dxU_f.transpose())*normal );
      PD += JxW*(ftauqp.dot(svqp));
      PDu+= JxW*(ftauqp.dot(Uqp));
      //cout << Traction_.transpose() << "  " << areas << endl;
    }
  }

  cout << "Traction Force = " << Traction_.transpose() << ". Area = " << areas
       << ". Power integral ftau*us = " << PD;

  double B1 = +0.5, B2 = +3.0, R = 1.0;
  double VDr = 16.0*pi*muu(0)*R*(B1*B1/3.0 + B2*B2/6.0);
  cout << ". Viscous Dissipation = " << VD << ". Numeric Disipation = " << ND;
  cout << ". Vicouss Dissipation - Power = " << VD + PD << endl;
  cout << ". Viscous total = " << VT << ". Power integral ftau*u = " << PDu << endl; //". Difference = " << PD+VT << endl;
  //cout << ". Exact VD = " << VDr;

  //save traction and power
  filf.open(forc,iostream::app);
  filf.precision(15);
  filf << current_time << /*" " << Traction_.transpose() <<*/ " " << PD << " ";
  filf.close();
  //save viscous dissipation
  filf.open(forc,iostream::app);
  filf.precision(15);
  filf << VD << " " << VD + PD << " " << ND << " " << PDu << endl;
  filf.close();

}

void AppCtx::pressureTimeCorrection(Vec &Vec_up_0, Vec &Vec_up_1, double a, double b) // p(n+1) = a*p(n+.5) + b* p(n)
{
  Vector    Uf(dim);
  Vector    X(dim);
  Tensor    R(dim,dim);
  int       dof;

  double P0, P1, P2;

  if (behaviors & BH_Press_grad_elim)
  {
    cout << "FIX ME: NOT SUPPORTED YET!!!!\n";
    throw;
  }

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    // press
    if (mesh->isVertex(&*point))
    {
      getNodeDofs(&*point,DH_UNKM,VAR_P,&dof);
      VecGetValues(Vec_up_1, 1, &dof, &P1);
      VecGetValues(Vec_up_0, 1, &dof, &P0);
      P2 = a*P1 + b*P0;
      VecSetValues(Vec_up_0, 1, &dof, &P2, INSERT_VALUES);
    }

  } // end point loop
  Assembly(Vec_up_0);
}

double AppCtx::getMaxVelocity()
{
  Vector     Uf(dim); // Ue := elastic velocity
  VectorXi   vtx_dofs_fluid(dim); // indices de onde pegar a velocidade
  double     Umax=0;

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    if (!mesh->isVertex(&*point))
        continue;
    dof_handler[DH_UNKM].getVariable(VAR_U).getVertexDofs(vtx_dofs_fluid.data(), &*point);
    //VecGetValues(Vec_up_1, dim, vtx_dofs_fluid.data(), Uf.data());
    Umax = max(Umax,Uf.norm());
  }
  return Umax;
}

double AppCtx::getMeshVolume()
{
  MatrixXd            x_coefs_c(nodes_per_cell, dim);
  MatrixXd            x_coefs_c_trans(dim, nodes_per_cell);
  Tensor              F_c(dim,dim), invF_c(dim,dim), invFT_c(dim,dim);
  VectorXi            cell_nodes(nodes_per_cell);
  double              Jx;
  //int                 tag;
  double              volume=0;

  cell_iterator cell = mesh->cellBegin();
  cell_iterator cell_end = mesh->cellEnd();
  for (; cell != cell_end; ++cell)
  {
    //tag = cell->getTag();

    mesh->getCellNodesId(&*cell, cell_nodes.data());
    mesh->getNodesCoords(cell_nodes.begin(), cell_nodes.end(), x_coefs_c.data());
    x_coefs_c_trans = x_coefs_c.transpose();

    for (int qp = 0; qp < n_qpts_err; ++qp)
    {
      F_c    = x_coefs_c_trans * dLqsi_err[qp];
      Jx     = F_c.determinant();
      volume += Jx * quadr_err->weight(qp);

    } // fim quadratura
    //cout << volume << "  ";
  } // end elementos
  return volume;
}

//void AppCtx::getOrderedCellIds(Cell const* cell, int *result)
//{
//  mesh->getCellNodesId(&*cell, cell_nodes.data());
//}

void AppCtx::getSolidVolume()
{
  VV.assign(n_solids,0.0);

//#ifdef FEP_HAS_OPENMP
//  FEP_PRAGMA_OMP(parallel default(none))
//#endif
  {
  MatrixXd            x_coefs_c(nodes_per_cell, dim);
  MatrixXd            x_coefs_c_trans(dim, nodes_per_cell);
  Tensor              F_c(dim,dim);
  VectorXi            cell_nodes(nodes_per_cell);
  double              Jx; //tvol;
  int                 tag, nod_id;
  Vector              Xqp(dim), Xqp3(Vector::Zero(3));
//  VectorXi            cell_nodes_tmp(nodes_per_cell);
  Tensor              F_c_curv(dim,dim);
  int                 tag_pt0, tag_pt1, tag_pt2, bcell;
  double const*       Xqpb;  //Coordinates at the master element \hat{X}
  Vector              Phi(dim), DPhi(dim), Dphi(dim), X0(dim), X2(dim), T0(dim), T2(dim), Xc(3), Vdat(3);
  bool                curvf = false;
  TensorXi            PerM3(TensorXi::Zero(3,3));  //Permutation matrix
  PerM3(0,1) = 1; PerM3(1,2) = 1; PerM3(2,0) = 1;

  const int tid = omp_get_thread_num();
  const int nthreads = omp_get_num_threads();
  cell_iterator cell = mesh->cellBegin(tid,nthreads);
  cell_iterator cell_end = mesh->cellEnd(tid,nthreads);
  for (; cell != cell_end; ++cell)
  {
    tag = cell->getTag();
    nod_id = is_in_id(tag, solidonly_tags);
    if (nod_id){
      //get nodal coordinates of the old and (permuted) new cell//////////////////////////////////////////////////
      mesh->getCellNodesId(&*cell, cell_nodes.data());
      if(is_curvt){
        //find good orientation for nodes in case of curved border element
        tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();  //cout << cell_nodes.transpose();
        tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
        tag_pt2 = mesh->getNodePtr(cell_nodes(2))->getTag();
        //test if the cell is acceptable (one curved side)
        bcell = is_in(tag_pt0,solidonly_tags)+is_in(tag_pt1,solidonly_tags)+is_in(tag_pt2,solidonly_tags);
        curvf = bcell==1;
        if (curvf){
          while (!is_in(tag_pt1, solidonly_tags)){
            //cell_nodes_tmp = cell_nodes;
            //cell_nodes(0) = cell_nodes_tmp(1);
            //cell_nodes(1) = cell_nodes_tmp(2);
            //cell_nodes(2) = cell_nodes_tmp(0);
            // TODO if P2/P1
            //cell_nodes(3) = cell_nodes_tmp(4);
            //cell_nodes(4) = cell_nodes_tmp(5);
            //cell_nodes(5) = cell_nodes_tmp(3);
            cell_nodes = PerM3*cell_nodes;  //cout << " permuted to " << cell_nodes.transpose();
            tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
          }
        }
      }  //cout << endl;
      ////////////////////////////////////////////////////////////////////////////////////////////////////
      mesh->getNodesCoords(cell_nodes.begin(), cell_nodes.end(), x_coefs_c.data());
      x_coefs_c_trans = x_coefs_c.transpose();

      if (curvf){
        X0(0) = x_coefs_c_trans(0,0);  X0(1) = x_coefs_c_trans(1,0);
        X2(0) = x_coefs_c_trans(0,2);  X2(1) = x_coefs_c_trans(1,2);
        //Xc << 8.0, 8.0, 0.0; //cout << cell_nodes.transpose() << endl << x_coefs_c << endl << endl;
        Xc = XG_0[nod_id-1];
        T0 = exact_tangent_ellipse(X0,Xc,0.0,RV[nod_id-1](0),RV[nod_id-1](1),dim);
        //T0 = exact_tangent_ellipse(X0,Xc,0.0,1.20,1.2,dim);  //cout << X0.transpose() << endl;
        T2 = exact_tangent_ellipse(X2,Xc,0.0,RV[nod_id-1](0),RV[nod_id-1](1),dim);
        //T2 = exact_tangent_ellipse(X2,Xc,0.0,1.20,1.2,dim);  //cout << X2.transpose() << endl;
        Vdat << RV[nod_id-1](0),RV[nod_id-1](1), 0.0; //container for R1, R2, theta
      }
      //tvol = 0.0;
      for (int qp = 0; qp < n_qpts_err; ++qp)
      {
        F_c = Tensor::Zero(dim,dim);  //Zero(dim,dim);
        if (curvf){//F_c_curv.setZero();
          Xqpb = quadr_err->point(qp);  //cout << Xqpb[0] << "   " << Xqpb[1] << endl;
          Phi = curved_Phi(Xqpb[1],X0,X2,Xc,Vdat,dim); //curved_Phi(Xqpb[1],X0,X2,-T0,-T2,dim);
          //cout << Phi.transpose() << endl;
          //cout << exact_ellipse(Xqpb[1],X0,X2,Xc,0.0,1.2,1.2,dim).transpose() << endl;
          //cout << ((1-Xqpb[0]-Xqpb[1])*x_coefs_c_trans.col(0) + Xqpb[0]*x_coefs_c_trans.col(1)
          //        + Xqpb[1]*x_coefs_c_trans.col(2) + (1-Xqpb[0]-Xqpb[1])*Phi).transpose() << endl;
          //cout << ((1-Xqpb[0]-Xqpb[1])*x_coefs_c_trans.col(0) + Xqpb[0]*x_coefs_c_trans.col(1)
          //          + Xqpb[1]*x_coefs_c_trans.col(2)).transpose() << endl;
          DPhi = Dcurved_Phi(Xqpb[1],X0,X2,Xc,Vdat,dim);
          //cout << DPhi.transpose() << endl;
          F_c_curv.col(0) = -Phi;
          F_c_curv.col(1) = -Phi + (1.0-Xqpb[0]-Xqpb[1])*DPhi;
          //cout << F_c_curv << endl;
          F_c = F_c_curv;
        }

        F_c   += x_coefs_c_trans * dLqsi_err[qp];// + curvf*F_c_curv;  //cout << dLqsi_err[qp];
        Jx     = F_c.determinant();
        if (is_axis){
          Xqp = x_coefs_c_trans * qsi_err[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
          Jx = Jx*2.0*pi*Xqp(0);
        }
        VV[nod_id-1] += Jx * quadr_err->weight(qp); //cout << VV[nod_id-1] << endl;
        //tvol = tvol + Jx * quadr_err->weight(qp);
      } // fim quadratura
      //cout << mesh->getCellId(&*cell) << "   " << tvol << endl;
    }
  } //cout << "approx area = " << VV[0] << ", area difference = " << abs(VV[0]-pi/2.0/*4.0/3.0*/) << endl << endl; // end elementos
  }
}

void AppCtx::getSolidCentroid()
{
  Vector XG(Vector::Zero(3));
  XG_0.assign(n_solids,XG);

//#ifdef FEP_HAS_OPENMP
//  FEP_PRAGMA_OMP(parallel default(none))
//#endif
  {
  MatrixXd            x_coefs_c(nodes_per_cell, dim);
  MatrixXd            x_coefs_c_trans(dim, nodes_per_cell);
  Tensor              F_c(dim,dim);
  VectorXi            cell_nodes(nodes_per_cell);
  double              Jx;
  int                 tag, nod_id;
  Vector              Xqp(dim), Xqp3(Vector::Zero(3));
//  VectorXi            cell_nodes_tmp(nodes_per_cell);
  Tensor              F_c_curv(dim,dim);
  int                 tag_pt0, tag_pt1, tag_pt2, bcell;
  double const*       Xqpb;  //coordinates at the master element \hat{X}
  Vector              Phi(dim), DPhi(dim), Dphi(dim), X0(dim), X2(dim), T0(dim), T2(dim), Xc(3), Vdat(3);
  bool                curvf = false;
  TensorXi            PerM3(TensorXi::Zero(3,3));  //Permutation matrix
  PerM3(0,1) = 1; PerM3(1,2) = 1; PerM3(2,0) = 1;

  const int tid = omp_get_thread_num();
  const int nthreads = omp_get_num_threads();
  cell_iterator cell = mesh->cellBegin(tid,nthreads);
  cell_iterator cell_end = mesh->cellEnd(tid,nthreads);
  for (; cell != cell_end; ++cell)
  {
    tag = cell->getTag();
    nod_id = is_in_id(tag, solidonly_tags);
    if (nod_id){
      mesh->getCellNodesId(&*cell, cell_nodes.data());

      if(is_curvt){
        //find good orientation for nodes in case of curved border element
        tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();
        tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
        tag_pt2 = mesh->getNodePtr(cell_nodes(2))->getTag();
        bcell = is_in(tag_pt0,solidonly_tags)
               +is_in(tag_pt1,solidonly_tags)
               +is_in(tag_pt2,solidonly_tags);  //test if the cell is acceptable (one curved side)
        curvf = bcell==1;// && is_curvt;
        if (curvf){
          while (!is_in(tag_pt1, solidonly_tags)){
            cell_nodes = PerM3*cell_nodes;
            tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
          }
        }
      }

      mesh->getNodesCoords(cell_nodes.begin(), cell_nodes.end(), x_coefs_c.data());
      x_coefs_c_trans = x_coefs_c.transpose();

      if (curvf){
        X0(0) = x_coefs_c_trans(0,0);  X0(1) = x_coefs_c_trans(1,0);
        X2(0) = x_coefs_c_trans(0,2);  X2(1) = x_coefs_c_trans(1,2);
        Xc << 8.0, 8.0, 0.0; //cout << cell_nodes.transpose() << endl << x_coefs_c << endl << endl;
        T0 = exact_tangent_ellipse(X0,Xc,0.0,1.20,1.2,dim);  //cout << X0.transpose() << endl;
        T2 = exact_tangent_ellipse(X2,Xc,0.0,1.20,1.2,dim);  //cout << X2.transpose() << endl;
        Vdat << 1.20, 1.20, 0.0;
      }

      for (int qp = 0; qp < n_qpts_err; ++qp)
      {
        F_c = Tensor::Zero(dim,dim);
        if (curvf){
          Xqpb = quadr_err->point(qp);  //cout << Xqpb[0] << "   " << Xqpb[1] << endl;
          Phi = curved_Phi(Xqpb[1],X0,X2,Xc,Vdat,dim); //curved_Phi(Xqpb[1],X0,X2,-T0,-T2,dim);
          DPhi = Dcurved_Phi(Xqpb[1],X0,X2,Xc,Vdat,dim);
          F_c_curv.col(0) = -Phi;
          F_c_curv.col(1) = -Phi + (1.0-Xqpb[0]-Xqpb[1])*DPhi;
          F_c = F_c_curv;
        }

        Xqp     = x_coefs_c_trans * qsi_err[qp];
        Xqp3(0) = Xqp(0); Xqp3(1) = Xqp(1); if (dim == 3) Xqp3(2) = Xqp(2);
        F_c    += x_coefs_c_trans * dLqsi_err[qp];// + curvf*F_c_curv;
        Jx      = F_c.determinant();
        if (is_axis){
          Jx = Jx*2.0*pi*Xqp(0);
        }
        XG_0[nod_id-1] += Xqp3 * Jx * quadr_err->weight(qp)/VV[nod_id-1]; //cout << XG_0[nod_id-1] << endl;
      } // fim quadratura
    }
  } // end elementos
  }
}

Vector AppCtx::getAreaMassCenterSolid(int sol_id, double &A){
  std::vector<Vector2d> XP;
  Vector Xp(dim);
  int tag, nod_id;
  int NS = NN_Solids[sol_id-1];
  VectorXi mapM_r(dim);
  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  //capture the boundary nodes
  for (; point != point_end; ++point){
	tag = point->getTag();
	nod_id = is_in_id(tag,flusoli_tags);
	if (nod_id == sol_id){
	  dof_handler[DH_UNKM].getVariable(VAR_U).getVertexDofs(mapM_r.data(),&*point);
      //VecGetValues(Vec_x_0, mapM_r.size(), mapM_r.data(), Xp.data());
      //VecGetValues(Vec_x_1, mapM_r.size(), mapM_r.data(), Xp.data());
	  point->getCoord(Xp.data(),dim);
      XP.push_back(Xp);
	}
  }
  XP.resize(NS);
  //ordering the point cloud XP
  std::vector<Vector2d> XPO;
  XPO = ConvexHull2d(XP);
  XPO.push_back(XPO[0]);
  //for (int j = 0; j < NN_Solids[sol_id-1]+1; j++)
//    cout << XPO[j].transpose() << "   ";
//  cout << endl;

  //area
  A = 0;//double A = 0;
  for (int i = 0; i < NS; i++){
    A = A + XPO[i](0)*XPO[i+1](1) - XPO[i+1](0)*XPO[i](1);
  }
  A = 0.5*A;  //cout << A << endl;
  //mass center
  Vector CM(dim);
  CM = Vector::Zero(2);
  for (int i = 0; i < NS; i++){
    CM(0) = CM(0) + (XPO[i](0) + XPO[i+1](0))*(XPO[i](0)*XPO[i+1](1) - XPO[i+1](0)*XPO[i](1));
	CM(1) = CM(1) + (XPO[i](1) + XPO[i+1](1))*(XPO[i](0)*XPO[i+1](1) - XPO[i+1](0)*XPO[i](1));
  }
  CM(0) = CM(0)/(6*A);
  CM(1) = CM(1)/(6*A);
  //cout << CM.transpose() << endl;
  return CM;
}

std::vector<Vector2d> AppCtx::ConvexHull2d(std::vector<Vector2d> & LI){
  int N = LI.size(), k = 0;
  std::vector<Vector2d> H(2*N);
  //lexicographically
  std::sort(LI.begin(),LI.end(),lessVector);
//  for (unsigned int j = 0; j < LI.size(); j++)
//	  cout << LI[j].transpose() << "   ";
//  cout << endl;
  // Build lower hull
  for (int i = 0; i < N; ++i) {
    while (k >= 2 && cross2d(H[k-2], H[k-1], LI[i]) <= 0) k--;
    H[k++] = LI[i];
  }
  // Build upper hull
  for (int i = N-2, t = k+1; i >= 0; i--) {
    while (k >= t && cross2d(H[k-2], H[k-1], LI[i]) <= 0) k--;
    H[k++] = LI[i];
  }

  H.resize(k);
  return H;
}

void AppCtx::calcHmean(double &hmean, double &hmin, double &hmax){
  VectorXi edge_nodes(3);
  Vector Xa(dim), Xb(dim);
  int n_edges=0;
  double hmn = 0, hmx = 0;
  hmean = 0;

  if (dim==2)
  //FEP_PRAGMA_OMP(parallel default(none) shared(hmean))
  {
    const int n_edges_total = mesh->numFacetsTotal();
    Facet const* edge(NULL);
    //double hlist[n_edges_total];

    //FEP_PRAGMA_OMP(for nowait)
    for (int a = 0; a < n_edges_total; ++a)
    {
      edge = mesh->getFacetPtr(a);
      if (edge->isDisabled())
        continue;

      mesh->getFacetNodesId(&*edge, edge_nodes.data());

      mesh->getNodePtr(edge_nodes[0])->getCoord(Xa.data(),dim);
      mesh->getNodePtr(edge_nodes[1])->getCoord(Xb.data(),dim);
      hmean += (Xa-Xb).norm(); //hlist[a] = (Xa-Xb).norm();
      ++n_edges;
    }
    //cout << "Here" << endl;
    //hmn = *min_element(hlist,hlist+n_edges_total);
    //hmx = *max_element(hlist,hlist+n_edges_total);
  }
  else
  if (dim==3)
  //FEP_PRAGMA_OMP(parallel default(none) shared(cout,hmean))
  {
    const int n_edges_total = mesh->numCornersTotal();
    Corner const* edge(NULL);
    //double hlist[n_edges_total];
    //FEP_PRAGMA_OMP(for nowait)
    for (int a = 0; a < n_edges_total; ++a)
    {
      edge = mesh->getCornerPtr(a);
      if (edge->isDisabled())
        continue;

      mesh->getCornerNodesId(&*edge, edge_nodes.data());

      mesh->getNodePtr(edge_nodes[0])->getCoord(Xa.data(),dim);
      mesh->getNodePtr(edge_nodes[1])->getCoord(Xb.data(),dim);
      hmean += (Xa-Xb).norm(); //hlist[a] = (Xa-Xb).norm();
      ++n_edges;
    }
    //hmn = *min_element(hlist,hlist+n_edges_total);
    //hmx = *max_element(hlist,hlist+n_edges_total);
  }
  hmean /= n_edges;
  hmin = hmn; hmax = hmx;
}

bool AppCtx::proxTest(MatrixXd &ContP, MatrixXd &ContW, double const INF){
  ContP = MatrixXd::Constant(n_solids,n_solids,INF);
  ContW = MatrixXd::Constant(n_solids,5,INF);  //5 tags for dirichlet

//  std::vector<bool> Contact(n_solids+1,false);
//  std::vector<double> Hmin(n_solids+1,0.0);
  VectorXi edge_nodes(3); // 3 nodes at most
  Vector Xa(dim), Xb(dim);
  const int n_edges_total = mesh->numFacetsTotal();
  Facet const* edge(NULL);
  int tag_a, tag_b, tag_e;
  int K, L;
  bool RepF = false;
  double hv = 0;

  //FEP_PRAGMA_OMP(for nowait)
  for (int a = 0; a < n_edges_total; ++a)
  {
    edge = mesh->getFacetPtr(a);
    if (edge->isDisabled())
      continue;

    mesh->getFacetNodesId(&*edge, edge_nodes.data());

    tag_a = mesh->getNodePtr(edge_nodes[0])->getTag();
    if (mesh->isVertex(mesh->getNodePtr(edge_nodes[1])))
      tag_b = mesh->getNodePtr(edge_nodes[1])->getTag();
    else
      tag_b = mesh->getNodePtr(edge_nodes[2])->getTag();
    tag_e = edge->getTag();
    //cout << tag_a << "  " << tag_b << "  " << tag_e << endl;

    if ((tag_a==tag_b) && (tag_b==tag_e))  //only edges outside the same type of domain
      continue;
    if ( (is_in(tag_a, fluidonly_tags) && !is_in(tag_a, dirichlet_tags)) ||
         (is_in(tag_b, fluidonly_tags) && !is_in(tag_b, dirichlet_tags)) )
      continue;
    if (is_in(tag_e, dirichlet_tags))                                  //avoid dirichlet edges
      continue;
    if (is_in(tag_a, dirichlet_tags) && is_in(tag_b, dirichlet_tags))  //avoid dirichlet edges
      continue;
    if (is_in(tag_e, flusoli_tags) || is_in(tag_e, slipvel_tags) || is_in(tag_e, solidonly_tags))
      continue;

    RepF = true;

    mesh->getNodePtr(edge_nodes[0])->getCoord(Xa.data(),dim);
    if (mesh->isVertex(mesh->getNodePtr(edge_nodes[1])))
      mesh->getNodePtr(edge_nodes[1])->getCoord(Xb.data(),dim);
    else
      mesh->getNodePtr(edge_nodes[2])->getCoord(Xb.data(),dim);
    hv = (Xa-Xb).norm();  //cout << Xa.transpose() << "   " << Xb.transpose() << endl;

    K = is_in_id(tag_a,flusoli_tags)+is_in_id(tag_a,slipvel_tags);  //K or L = 0, means the point is wall
    L = is_in_id(tag_b,flusoli_tags)+is_in_id(tag_b,slipvel_tags);
    if ((K != 0) && (L != 0)){
      if (hv < ContP(K-1,L-1)){
        ContP(K-1,L-1) = hv;
        ContP(L-1,K-1) = hv;
      }  //saves the smallest distance
    }
    else if (K == 0){
      K = is_in_id(tag_a,dirichlet_tags);
      cout << K << "  " << mesh->getPointId(mesh->getNodePtr(edge_nodes[0]))
                << "  " << mesh->getPointId(mesh->getNodePtr(edge_nodes[1])) << endl;
      if (hv < ContW(L-1,K-1)){
        ContW(L-1,K-1) = hv;
      }
    }
    else if (L == 0){
      L = is_in_id(tag_b,dirichlet_tags);
      cout << L << "  " << mesh->getPointId(mesh->getNodePtr(edge_nodes[0]))
                << "  " << mesh->getPointId(mesh->getNodePtr(edge_nodes[1])) << endl;
      if (hv < ContW(K-1,L-1)){
        ContW(K-1,L-1) = hv;
      }
    }
  }//end for edges
  return RepF;
}

void AppCtx::forceDirichlet(){
  Vector    U1(dim);
  Vector    X1(dim);
  VectorXi  u_dofs(dim);
  int tag;
  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for ( ; point != point_end; ++point)
  {
    tag = point->getTag();
    if (is_in(tag,dirichlet_tags)){
      point->getCoord(X1.data(),dim);
      getNodeDofs(&*point, DH_UNKM, VAR_U, u_dofs.data());
      U1 = u_exact(X1, current_time+dt, tag);
      VecSetValues(Vec_ups_1, dim, u_dofs.data(), U1.data(), INSERT_VALUES);
    }
  }
  Assembly(Vec_ups_1);
}

PetscErrorCode AppCtx::updateSolidVel()//FIXME: deprecated function
{
  VectorXi  dofs(dim), mapM_r(dim);
  VectorXi  dofs_fs(LZ);
  Vector    X(dim);
  Vector    Uf(dim), Zf(LZ), Vs(dim);
  Vector3d  Xg;
  int       nod_id, nod_is, nod_vs, nodsum;
  int       tag;
  vector<bool>   SV(n_solids,false);  //solid visited history

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    tag = point->getTag();
    // vel
    nod_id = is_in_id(tag,flusoli_tags);
    nod_is = is_in_id(tag,solidonly_tags);
    nod_vs = is_in_id(tag,slipvel_tags);
    nodsum = nod_id+nod_is+nod_vs;
    if (nodsum){
      for (int l = 0; l < LZ; l++){
        dofs_fs(l) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + l;
      }
      dof_handler[DH_UNKM].getVariable(VAR_U).getVertexDofs(mapM_r.data(),&*point);
      VecGetValues(Vec_x_1, mapM_r.size(), mapM_r.data(), X.data()); //point->getCoord(X.data(),dim);
      VecGetValues(Vec_ups_1, LZ, dofs_fs.data(), Zf.data());
      Uf = SolidVel(X, XG_1[nodsum-1], Zf, dim);
      if (nod_vs){  //ojo antes solo nod_vs   //+nod_id
        getNodeDofs(&*point, DH_MESH, VAR_M, dofs.data());
        VecGetValues(Vec_slipv_1, dim, dofs.data(), Vs.data());
        Uf = Uf + Vs;  //cout << tag << "  " << X(0)-3 << " " << X(1)-3<< "  " << Uf.transpose() << endl;
      }
      //cout << dofs_fs.transpose() << ", " << X.transpose() << ", " << XG_0[nod_id+nod_is-1].transpose() << ", " << Zf.transpose() << ", "<< Uf.transpose() << endl;
      getNodeDofs(&*point,DH_UNKM,VAR_U,dofs.data());
      VecSetValues(Vec_ups_1, dim, dofs.data(), Uf.data(), INSERT_VALUES);
    }
  } // end point loop
  Assembly(Vec_ups_1);
  //char buf1[18], buf2[10]; //sprintf(buf1,"matrizes/sol%d.m",time_step); sprintf(buf2,"solm%d",time_step); //View(Vec_ups_1, buf1, buf2);
  PetscFunctionReturn(0);  //cout << endl;
}

PetscErrorCode AppCtx::moveSolidDOFs(double const stheta)
{
  VectorXi    dofs(LZ), dof(1);
  Vector      U0(Vector::Zero(3)), U1(Vector::Zero(3)), XG_temp(Vector::Zero(3));
  Vector      Omega0(Vector::Zero(3)), Omega1(Vector::Zero(3));
  Vector      Z0(Vector::Zero(LZ)), Z1(Vector::Zero(LZ));
  Matrix3d    Id3(Matrix3d::Identity(3,3)), Qtmp;
  double      theta_temp, omega0, omega1, modes0, modes1;
  //cout << "here" << endl;
  for (int s = 0; s < n_solids; s++){

    for (int l = 0; l < LZ; l++){
      dofs(l) = n_unknowns_u + n_unknowns_p + LZ*s + l;
    }
    VecGetValues(Vec_ups_0, LZ, dofs.data(), Z0.data());
    VecGetValues(Vec_ups_1, LZ, dofs.data(), Z1.data());
    //dof(0) = n_unknowns_u + n_unknowns_p + LZ*s + dim;
    //VecGetValues(Vec_ups_0, 1, dof.data(), omega0.data());
    //VecGetValues(Vec_ups_1, 1, dof.data(), omega1.data());
    U0.head(dim) = Z0.head(dim); omega0 = Z0(LZ-(1+n_modes)); modes0 = Z0(LZ-1);
    U1.head(dim) = Z1.head(dim); omega1 = Z1(LZ-(1+n_modes)); modes1 = Z1(LZ-1);
    if (dim == 2){Omega0(2) = omega0;     Omega1(2) = omega1;    }
    else{         Omega0    = Z0.tail(3); Omega1    = Z1.tail(3);}

    if (is_mr_qextrap && time_step > 0){
      XG_1[s]    = XG_0[s]    + stheta*dt*U1;  //equiv. to XG_1[s] = (1.0+stheta)*XG_0[s] - stheta*XG_m1[s];
      theta_1[s] = theta_0[s] + stheta*dt*omega1;  //equiv. to theta_1[s] = (1.0+stheta)*theta_0[s] - stheta*theta_m1[s];
      modes_1[s] = modes_0[s] + stheta*dt*modes1;  //equiv. to theta_1[s] = (1.0+stheta)*theta_0[s] - stheta*theta_m1[s];
      //Qtmp = Id3 - stheta*dt*SkewMatrix(Z1.tail(3));
      //invert(Qtmp,dim); Q_1[s] = Qtmp*Q_0[s];
      Qtmp = Q_0[s] + dt*stheta*SkewMatrix(Omega1)*Q_0[s];
      ProjOrtMatrix(Qtmp, 0/*1e4*/, 1e-8, dim);
      Q_1[s] = Qtmp;
    }
    else if (is_mr_ab && time_step > 0){
      XG_1[s]    = XG_0[s]    + dt*((1.0+stheta)*U1 - stheta*U0);
      theta_1[s] = theta_0[s] + dt*((1.0+stheta)*omega1 - stheta*omega0);
      modes_1[s] = modes_0[s] + dt*((1.0+stheta)*modes1 - stheta*modes0);
      //Qtmp = Id3 - stheta*dt*SkewMatrix(Z1.tail(3));
      //invert(Qtmp,dim); Q_1[s] = Qtmp*Q_0[s];
      Qtmp = Q_0[s] + dt*((1.0+stheta)*SkewMatrix(Omega1)*Q_1[s]
                              -stheta *SkewMatrix(Omega0)*Q_m1[s]);//Q_m1 has the time_step=n-2 info
      ProjOrtMatrix(Qtmp, 0/*1e4*/, 1e-8, dim);
      Q_1[s] = Qtmp;
    }
    else if (is_bdf3 && time_step > 1){
      XG_temp   = XG_1[s];
      XG_1[s]   = (18./11.)*XG_1[s] - (9./11.)*XG_0[s] + (2./11.)*XG_m1[s] + (6./11.)*dt*U0;
      XG_m1[s] = XG_0[s];
      XG_0[s]   = XG_temp;
      theta_temp   = theta_1[s];
      theta_1[s]   = (18./11.)*theta_1[s] - (9./11.)*theta_0[s] + (2./11.)*theta_m1[s] + (6./11.)*dt*omega0;
      theta_m1[s] = theta_0[s];
      theta_0[s]   = theta_temp;
    }
    else if (is_bdf2 && time_step > 0){
      if (is_bdf2_bdfe){
        XG_temp = XG_1[s];
        XG_1[s] = (4./3.)*XG_1[s] - (1./3.)*XG_0[s] + (2./3.)*dt*(stheta*U1 + (1.-stheta)*U0);
        XG_0[s] = XG_temp;
        theta_temp = theta_1[s];
        theta_1[s] = (4./3.)*theta_1[s] - (1./3.)*theta_0[s] + (2./3.)*dt*(stheta*omega1 + (1.-stheta)*omega0);
        theta_0[s] = theta_temp;
      }
      else if (is_bdf2_ab){
        XG_0[s] = XG_1[s];
        XG_1[s] = dt*(stheta*U1 + (1.-stheta)*U0) + XG_0[s];
        //XG_0[s] = XG_temp;
        theta_0[s] = theta_1[s];
        theta_1[s] = dt*(stheta*omega1 + (1.-stheta)*omega0) + theta_0[s];
        //theta_0[s] = theta_temp;
      }
    }
    else{  //for MR-AB and basic, and for all at time = t0
      if (time_step == 0 || (is_bdf3 && time_step == 1)){
        XG_1[s]    = XG_0[s]    + dt*U1;
        theta_1[s] = theta_0[s] + dt*omega1;
        modes_1[s] = modes_0[s] + dt*modes1;
        if (dim == 3){//TODO
          //Qtmp = Id3 - 0.0*dt*SkewMatrix(Z1.tail(3));
          //invert(Qtmp,dim); Q_1[s] = Qtmp*Q_0[s];
          Qtmp = Q_0[s] + dt*SkewMatrix(Omega1)*Q_0[s];
          ProjOrtMatrix(Qtmp, 0/*1e4*/, 1e-8, dim);
          Q_1[s] = Qtmp;
        }
      }
      else{
        XG_0[s]    = XG_1[s];
        XG_1[s]    = dt*(stheta*U1     + (1.-stheta)*U0)     + XG_0[s]; //XG_0[s] = XG_temp;
        theta_0[s] = theta_1[s];
        theta_1[s] = dt*(stheta*omega1 + (1.-stheta)*omega0) + theta_0[s]; //theta_0[s] = theta_temp;
        modes_0[s] = modes_1[s];
        modes_1[s] = dt*(stheta*modes1 + (1.-stheta)*modes0) + modes_0[s];
      }
    }
    //cout << theta_0[s] << "   " << theta_1[s] << "     " << omega0(0) << "   " << omega1(0) << endl;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::velNoSlip(Vec const& Vec_uzp, Vec const& Vec_sv, Vec &Vec_ups_ns)
{
  int tag;
  VectorXi dofs(dim), dofs_mesh(dim);
  Vector   U(dim), Uns(dim);

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)  //to calculate Vec_v_mid at each point (initial guess)
  {
    tag = point->getTag();
    if (is_in(tag,slipvel_tags)){
      getNodeDofs(&*point, DH_UNKM, VAR_U, dofs.data());
      getNodeDofs(&*point, DH_MESH, VAR_M, dofs_mesh.data());
      VecGetValues(Vec_uzp,  dim, dofs.data(), U.data());
      VecGetValues(Vec_sv,  dim, dofs_mesh.data(), Uns.data());
      Uns = U - Uns;
      VecSetValues(Vec_ups_ns, dim, dofs.data(), Uns.data(), INSERT_VALUES);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::plotFiles(int step)
{
  double  *q_array;
  double  *nml_array;
  double  *v_array, *vs_array, *rho_array, *fd_array, *Ftau_array, *tg_array, *vm_array;
  if (step == 0){
    VecGetArray(Vec_ups_0, &q_array);}  //VecGetArray(Vec_up_0, &q_array);
  else{
    VecGetArray(Vec_ups_1, &q_array);}
  VecGetArray(Vec_normal, &nml_array);
  VecGetArray(Vec_v_mid, &v_array);
  if (is_sfip){
    if (step == 0){
      VecGetArray(Vec_slipv_0, &vs_array);}
    else{
      VecGetArray(Vec_slipv_1, &vs_array);}
    VecGetArray(Vec_fdis_0, &fd_array);
    VecGetArray(Vec_ftau_0, &Ftau_array);
    VecGetArray(Vec_tangent, &tg_array);
    VecGetArray(Vec_metav_0, &vm_array);
  }
  if (is_sslv) VecGetArray(Vec_slip_rho, &rho_array);
  vtk_printer.writeVtk();

  /* ---- nodes data ---- */
  vtk_printer.addNodeVectorVtk("u", GetDataVelocity(q_array, *this));
  vtk_printer.addNodeVectorVtk("n", GetDataNormal(nml_array, *this));
  vtk_printer.addNodeVectorVtk("v", GetDataMeshVel(v_array, *this));
  if (is_sfip){
    vtk_printer.addNodeVectorVtk("vs", GetDataMeshVel(vs_array, *this));
    vtk_printer.addNodeVectorVtk("fd", GetDataMeshVel(fd_array, *this));
    vtk_printer.addNodeVectorVtk("ftau", GetDataMeshVel(Ftau_array, *this));
    vtk_printer.addNodeVectorVtk("tg", GetDataMeshVel(tg_array, *this));
    vtk_printer.addNodeVectorVtk("vm", GetDataMeshVel(vs_array, *this));
  }
  if (is_sslv) vtk_printer.addNodeScalarVtk("rho", GetDataSlipVel(rho_array, *this));
  vtk_printer.printPointTagVtk();

  if (!shape_psi_c->discontinuous())
    vtk_printer.addNodeScalarVtk("p", GetDataPressure(q_array, *this));
  else
    vtk_printer.addCellScalarVtk("p", GetDataPressCellVersion(q_array, *this));

  vtk_printer.addCellIntVtk("cell_tag", GetDataCellTag(*this));

  //vtk_printer.printPointTagVtk("point_tag");
  if (step == 0){
    VecRestoreArray(Vec_ups_0, &q_array);}  //VecRestoreArray(Vec_up_0, &q_array);
  else{
    VecRestoreArray(Vec_ups_1, &q_array);}
  VecRestoreArray(Vec_normal, &nml_array);
  VecRestoreArray(Vec_v_mid, &v_array);
  if (is_sfip){
    if (step == 0){
      VecRestoreArray(Vec_slipv_0, &vs_array);}
    else{
      VecRestoreArray(Vec_slipv_1, &vs_array);}
    VecRestoreArray(Vec_fdis_0, &fd_array);
    VecRestoreArray(Vec_ftau_0, &Ftau_array);
    VecRestoreArray(Vec_tangent, &tg_array);
    VecRestoreArray(Vec_metav_0, &vm_array);
  }
  if (is_sslv) VecRestoreArray(Vec_slip_rho, &rho_array);

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::plotFilesNT()
{
  double  *q_array;
  double  *nml_array;
  double  *v_array, *vs_array, *rho_array, *fd_array, *Ftau_array, *tg_array;
  VecGetArray(Vec_ups_1, &q_array);  //VecGetArray(Vec_up_1, &q_array);
  VecGetArray(Vec_normal, &nml_array);
  VecGetArray(Vec_v_mid, &v_array);
  if (is_sfip){
    VecGetArray(Vec_slipv_1, &vs_array);
    VecGetArray(Vec_fdis_0, &fd_array);
    VecGetArray(Vec_ftau_0, &Ftau_array);
    VecGetArray(Vec_tangent, &tg_array);
  }
  if (is_sslv) VecGetArray(Vec_slip_rho, &rho_array);
  vtk_printer.writeVtk();

  /* ---- nodes data ---- */
  vtk_printer.addNodeVectorVtk("u", GetDataVelocity(q_array, *this));
  vtk_printer.addNodeVectorVtk("n", GetDataNormal(nml_array, *this));
  vtk_printer.addNodeVectorVtk("v", GetDataMeshVel(v_array, *this));
  if (is_sfip){
    vtk_printer.addNodeVectorVtk("vs", GetDataMeshVel(vs_array, *this));
    vtk_printer.addNodeVectorVtk("fd", GetDataMeshVel(fd_array, *this));
    vtk_printer.addNodeVectorVtk("ftau", GetDataMeshVel(Ftau_array, *this));
    vtk_printer.addNodeVectorVtk("tg", GetDataMeshVel(tg_array, *this));
  }
  if (is_sslv) vtk_printer.addNodeScalarVtk("rho", GetDataSlipVel(rho_array, *this));
  vtk_printer.printPointTagVtk();

  if (!shape_psi_c->discontinuous())
    vtk_printer.addNodeScalarVtk("p", GetDataPressure(q_array, *this));
  else
    vtk_printer.addCellScalarVtk("p", GetDataPressCellVersion(q_array, *this));

  vtk_printer.addCellIntVtk("cell_tag", GetDataCellTag(*this));

  //vtk_printer.printPointTagVtk("point_tag");
  VecRestoreArray(Vec_ups_1, &q_array);  //VecRestoreArray(Vec_up_1, &q_array);
  VecRestoreArray(Vec_normal, &nml_array);
  VecRestoreArray(Vec_v_mid, &v_array);
  if (is_sfip){
    VecRestoreArray(Vec_slipv_1, &vs_array);
    VecRestoreArray(Vec_fdis_0, &fd_array);
    VecRestoreArray(Vec_ftau_0, &Ftau_array);
    VecRestoreArray(Vec_tangent, &tg_array);
  }
  if (is_sslv) VecRestoreArray(Vec_slip_rho, &rho_array);

  PetscFunctionReturn(0);
}

void AppCtx::getSolidInertiaTensor()
{
  Tensor Z(3,3);
  Z.setZero();
  InTen.assign(n_solids,Z);

//#ifdef FEP_HAS_OPENMP
//  FEP_PRAGMA_OMP(parallel default(none))
//#endif
  {
  MatrixXd            x_coefs_c(nodes_per_cell, dim);
  MatrixXd            x_coefs_c_trans(dim, nodes_per_cell);
  Tensor              F_c(dim,dim);
  VectorXi            cell_nodes(nodes_per_cell);
  double              Jx, rho;
  int                 tag, nod_id, delta_ij;
  Vector              Xqp(dim), Xqp3(Vector::Zero(3));
  VectorXi            cell_nodes_tmp(nodes_per_cell);
  Tensor              F_c_curv(dim,dim);
  int                 tag_pt0, tag_pt1, tag_pt2, bcell;
  double const*       Xqpb;  //coordonates at the master element \hat{X}
  Vector              Phi(dim), DPhi(dim), Dphi(dim), X0(dim), X2(dim), T0(dim), T2(dim), Xc(3), Vdat(3);
  bool                curvf = false;
  TensorXi            PerM3(TensorXi::Zero(3,3));  //Permutation matrix
  PerM3(0,1) = 1; PerM3(1,2) = 1; PerM3(2,0) = 1;

  const int tid = omp_get_thread_num();
  const int nthreads = omp_get_num_threads();
  cell_iterator cell = mesh->cellBegin(tid,nthreads);
  cell_iterator cell_end = mesh->cellEnd(tid,nthreads);
  for (; cell != cell_end; ++cell)
  {
    tag = cell->getTag();
    nod_id = is_in_id(tag, solidonly_tags);
    if (nod_id){
      mesh->getCellNodesId(&*cell, cell_nodes.data());

      if(is_curvt){
        //find good orientation for nodes in case of curved border element
        tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();
        tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
        tag_pt2 = mesh->getNodePtr(cell_nodes(2))->getTag();
        bcell = is_in(tag_pt0,solidonly_tags)
               +is_in(tag_pt1,solidonly_tags)
               +is_in(tag_pt2,solidonly_tags);  //test if the cell is acceptable (one curved side)
        curvf = bcell==1 && is_curvt;
        if (curvf){
          while (!is_in(tag_pt1, solidonly_tags)){
            //cell_nodes_tmp = cell_nodes;
            //cell_nodes(0) = cell_nodes_tmp(1);
            //cell_nodes(1) = cell_nodes_tmp(2);
            //cell_nodes(2) = cell_nodes_tmp(0);
            // TODO if P2/P1
            //cell_nodes(3) = cell_nodes_tmp(4);
            //cell_nodes(4) = cell_nodes_tmp(5);
            //cell_nodes(5) = cell_nodes_tmp(3);
            cell_nodes = PerM3*cell_nodes;
            tag_pt1 = mesh->getNodePtr(cell_nodes(1))->getTag();
          }
        }
      }

      mesh->getNodesCoords(cell_nodes.begin(), cell_nodes.end(), x_coefs_c.data());
      x_coefs_c_trans = x_coefs_c.transpose();
      rho = MV[nod_id-1]/VV[nod_id-1];

      if (curvf){
        X0(0) = x_coefs_c_trans(0,0);  X0(1) = x_coefs_c_trans(1,0);
        X2(0) = x_coefs_c_trans(0,2);  X2(1) = x_coefs_c_trans(1,2);
        Xc << 8.0, 8.0, 0.0; //cout << cell_nodes.transpose() << endl << x_coefs_c << endl << endl;
        T0 = exact_tangent_ellipse(X0,Xc,0.0,1.20,1.2,dim);  //cout << X0.transpose() << endl;
        T2 = exact_tangent_ellipse(X2,Xc,0.0,1.20,1.2,dim);  //cout << X2.transpose() << endl;
        Vdat << 1.20, 1.20, 0.0;
      }

      for (int qp = 0; qp < n_qpts_err; ++qp)
      {
        F_c = Tensor::Zero(dim,dim);
        if (curvf){
          Xqpb = quadr_err->point(qp);  //cout << Xqpb[0] << "   " << Xqpb[1] << endl;
          Phi = curved_Phi(Xqpb[1],X0,X2,Xc,Vdat,dim); //curved_Phi(Xqpb[1],X0,X2,-T0,-T2,dim);
          DPhi = Dcurved_Phi(Xqpb[1],X0,X2,Xc,Vdat,dim);
          F_c_curv.col(0) = -Phi;
          F_c_curv.col(1) = -Phi + (1.0-Xqpb[0]-Xqpb[1])*DPhi;
          F_c = F_c_curv;
        }

        Xqp     = x_coefs_c_trans * qsi_err[qp];
        Xqp3(0) = Xqp(0); Xqp3(1) = Xqp(1); if (dim == 3) Xqp3(2) = Xqp(2);
        F_c    += x_coefs_c_trans * dLqsi_err[qp];// + curvf*F_c_curv;
        Jx      = F_c.determinant();
        if (is_axis){
          Jx = Jx*2.0*pi*Xqp(0);
        }

        for (int i = 0; i < 3; i++){
          for (int j = 0; j < 3; j++){
            delta_ij = i == j;
            InTen[nod_id-1](i,j) += rho*( (Xqp3-XG_0[nod_id-1]).squaredNorm()*delta_ij
                                         -(Xqp3(i)-XG_0[nod_id-1](i))*(Xqp3(j)-XG_0[nod_id-1](j))
                                        ) * Jx * quadr_err->weight(qp);
          } // end i
        } //end j
      } // end quadrature
    }
  } // end elements
  }
}

void AppCtx::getFromBSV() //Body Slip Velocity
{
  int                 tag, sign_;
  bool                is_slipvel, is_fsi;
  VectorXi            mapM_f(dim*nodes_per_facet);
  MatrixXd            x_coefs_f(n_dofs_v_per_facet/dim, dim), sv_coefs_f(n_dofs_v_per_facet/dim, dim);
  Tensor              F_f(dim,dim-1), fff_f(dim-1,dim-1);
  Vector              Xqp(dim), Xqp3(Vector::Zero(3)), Nr(dim), Vs(dim), TVsol(dim), TWsol(3), XG2(dim), Tg(dim);
  //const double        Pi = 3.141592653589793;
  double              Jx;

  Vsol = Vector3d::Zero(3); TVsol = Vector2d::Zero(2);
  Wsol = Vector3d::Zero(3); TWsol = Vector3d::Zero(3);

  facet_iterator facet = mesh->facetBegin();
  facet_iterator facet_end = mesh->facetEnd();  // the next if controls the for that follows

  if (slipvel_tags.size() != 0 || flusoli_tags.size() != 0)
  for (; facet != facet_end; ++facet)
  {
    tag = facet->getTag();
    is_slipvel = is_in_id(tag, slipvel_tags);
    is_fsi     = is_in_id(tag, flusoli_tags);
    if(!is_slipvel && !is_fsi)
      continue;

    dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(mapM_f.data(), &*facet);  //cout << mapM_f << endl;
    VecGetValues(Vec_x_0, mapM_f.size(), mapM_f.data(), x_coefs_f.data());
    VecGetValues(Vec_slipv_0, mapM_f.size(), mapM_f.data(), sv_coefs_f.data());

    // fix orientation in the case where the gas phase isn't passive
    {
      sign_ = 1;
      cell_handler cc = mesh->getCell(facet->getIncidCell());
      cell_handler oc = mesh->getCell(cc->getIncidCell(facet->getPosition()));
      if ( oc.isValid()  )
        if ( oc->getTag() > cc->getTag() )
          sign_ = -1;
    }

    // Quadrature
    for (int qp = 0; qp < n_qpts_facet; ++qp)
    {
      Xqp     = x_coefs_f.transpose() * qsi_f[qp];    //(2 because is the length of the master edge see shape_functions.cpp in fepicpp src)
      Xqp3(0) = Xqp(0); Xqp3(1) = Xqp(1); if (dim == 3) Xqp3(2) = Xqp(2);
      F_f     = x_coefs_f.transpose() * dLqsi_f[qp];  //cout << x_coefs_f.transpose() << endl;// << dLqsi_f[qp] << endl << F_f << endl <<
      fff_f   = F_f.transpose()*F_f;                  //cout << fff_f << endl << endl;
      Jx      = sqrt(fff_f.determinant());            //cout << Jx << endl << endl;//  Jx is 2times the lenght of the edge...
      if (is_axis){
        Jx = Jx*2.0*pi*Xqp(0);
      }

      //Nr      = Xqp;
      //Nr.normalize();  cout << Nr.transpose() << "   ";
      //Nr      = cross(F_f.col(0), F_f.col(1));
      Nr(0) = x_coefs_f.transpose()(1,1)-x_coefs_f.transpose()(1,0);
      Nr(1) = x_coefs_f.transpose()(0,0)-x_coefs_f.transpose()(0,1);
      Nr *= sign_;
      Nr.normalize();  //cout << Nr.transpose() << endl;
      Tg(0) = x_coefs_f.transpose()(0,1)-x_coefs_f.transpose()(0,0);
      Tg(1) = x_coefs_f.transpose()(1,1)-x_coefs_f.transpose()(1,0);

      Vs = SlipVel(Xqp, XG_0[is_slipvel+is_fsi-1], Nr, dim, tag, theta_ini[is_slipvel+is_fsi-1], 0.0, 0.0, current_time,
                   Q_0[is_slipvel+is_fsi-1], thetaDOF);
      //cout << Vs.dot(Nr) << "   " << Vs.dot(Xqp) << "   " << endl;
      //cout << Xqp.transpose() << "   " << Nr.transpose() << "   " << Tg.transpose() << "   " << Vs.transpose() << endl << endl;
      //cout << x_coefs_f.transpose() << endl << endl;
      //cout << Nr.dot(Tg)  << "   " << Nr.dot(Vs) << endl;//cout << XG_0[is_slipvel+is_fsi-1] << endl;

      if (dim == 2){
        XG2(0) = XG_0[is_slipvel+is_fsi-1](0); XG2(1) = XG_0[is_slipvel+is_fsi-1](1);
        TVsol += Nr.dot(Xqp-XG2) * Vs * Jx * quadr_facet->weight(qp) / (VV[is_slipvel+is_fsi-1]);
        //cout << quadr_facet->weight(qp) << endl;
        //cout << Nr.dot(Xqp-XG2) << "   " << Vs.transpose() << "   " << VV[is_slipvel+is_fsi-1] << endl << endl;
        //cout << TVsol.transpose() << endl;
        TWsol(2) += /*cross(Nr,Vs)*/(Nr(0)*Vs(1)-Nr(1)*Vs(0)) * Jx *
                                     quadr_facet->weight(qp) / (RV[is_slipvel+is_fsi-1](0)*RV[is_slipvel+is_fsi-1](0)*RV[is_slipvel+is_fsi-1])(0);
      }
      else{
        TVsol += Vs * Jx * quadr_facet->weight(qp) / (RV[is_slipvel-1](0)*RV[is_slipvel-1](0));
        TWsol += cross(Nr,Vs) * Jx * quadr_facet->weight(qp) / (RV[is_slipvel-1](0)*RV[is_slipvel-1](0)*RV[is_slipvel-1](0));
      }

    }
  }
  if (dim == 2){
    Vsol(0) = -(1.0/2.0)*TVsol(0);
    Vsol(1) = -(1.0/2.0)*TVsol(1);
    Wsol = -(1.0/(2.0*pi))*TWsol;
  }
  else{
    Vsol = -(1.0/(4.0*pi))*TVsol;
    Wsol = -(3.0/(8.0*pi))*TWsol;
  }
  cout << "Real Velocities:" << endl
       << "Translation = " << Vsol.transpose() << ", Rotation = " << Wsol.transpose() <<  endl;
}

Vector AppCtx::u_exacta(Vector const& X, double t, int tag)
{
  double x = X(0);
  double y = X(1);
  Vector v(Vector::Zero(X.size()));
  double r  = sqrt(x*x+y*y);
  double uth = 0.0;//, uthn = 0.0;
  int id = 0;
  double w2 = 2.0;
  Vector nor(Vector::Zero(X.size()));
  Vector tau(Vector::Zero(X.size()));
  //double theta = 0.0;
  if ( false && t == 0 ){
    if (tag == 1){
      v(0) = -w2*y;
      v(1) =  w2*x;
    }
  }

  if (is_axis){
    if (false && t > 0){
      for (int i = 0; i < 702; i++){
        if (r <= RR[i]) break;
        id++;
      }
      uth = UU[id]; //if (id < 301) {uthn = UU[id+1];}
      //cout << tag << "  " << x << "  " << y << "  " << r << "  " << uth << endl;
      //uth = (uth+uthn)/2;
      v(0) = uth*(-y/r);
      v(1) = uth*(x/r);
    }
    if (true)
    {
      Vector Xc(2), Y(2);
      double rb, thetab;
      double B1 = +0.5, B2 = +3.0, R = 1.0;
      Xc << XG_0[0](0), XG_0[0](1);
      Y = X - Xc;
      x = Y(0); y = Y(1);
      rb  = sqrt(x*x+y*y);
      nor = Y/rb; tau(0) = +nor(1); tau(1) = -nor(0);
      thetab = -atan2(Y(1),Y(0))+pi/2.0;
      v = ( (2.0/3.0)*B1*cos(thetab)*R*R*R/(rb*rb*rb)
          + (1.0/2.0)*B2*(R*R*R*R/(rb*rb*rb*rb) - R*R/(rb*rb))*(3.0*cos(thetab)*cos(thetab)-1.0) )*nor
         +( (1.0/3.0)*B1*sin(thetab)*R*R*R/(rb*rb*rb)
          +           B2*(R*R*R*R/(rb*rb*rb*rb)              )*sin(thetab)*cos(thetab)           )*tau;
    }
  }
  if (false && dim == 3){
    Vector Xc(3), Y(3);
    double rb, thetab;
    Xc = XG_0[0];
    Y = X - Xc;
    x = Y(0); y = Y(1);
    double z = Y(2);
    rb  = sqrt(x*x+y*y+z*z);
    nor = Y/rb; tau(0) = +nor(1); tau(1) = -nor(0);
    thetab = acos(z/rb);
    //thetab = atan2(Y(1),Y(0))+pi/2.0;
    v = cos(thetab)/(3.0*rb*rb*rb)*nor + sin(thetab)/(6.0*rb*rb*rb)*tau;
  }

  return v;
}

Tensor AppCtx::grad_u_exacta(Vector const& X, double t, int tag)
{
  double r, z, /*rb,*/ thetab;
  Tensor dxU(Tensor::Zero(dim,dim));
  Vector Xc(2), Y(2);
  if (t == -1 && tag == -1){}
  if (is_sfip && dim == 2){
    Xc << XG_0[0](0), XG_0[0](1);
    Y = X - Xc;
    r = Y(0); z = Y(1);
    //rb  = sqrt(r*r+z*z);
    thetab = atan2(Y(1),Y(0))+pi/2.0;
    dxU(0,0) = -(1.0/3.0)*sin(thetab) * (-z)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (-r)
               +(1.0/3.0)*cos(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*r) * (-r)
               +(1.0/3.0)*cos(thetab) *                          pow(r*r + z*z,-2) *                (-1)
               +(1.0/6.0)*cos(thetab) * (-z)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (z)
               +(1.0/6.0)*sin(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*r) * (z);

    dxU(1,0) = -(1.0/3.0)*sin(thetab) * (-z)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (-z)
               +(1.0/3.0)*cos(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*r) * (-z)
               +(1.0/6.0)*cos(thetab) * (-z)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (-r)
               +(1.0/6.0)*sin(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*r) * (-r)
               +(1.0/6.0)*sin(thetab) *                          pow(r*r + z*z,-2) *                (-1);

    dxU(0,1) = -(1.0/3.0)*sin(thetab) * ( r)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (-r)
               +(1.0/3.0)*cos(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*z) * (-r)
               +(1.0/6.0)*cos(thetab) * ( r)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (z)
               +(1.0/6.0)*sin(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*z) * (z)
               +(1.0/6.0)*sin(thetab) *                          pow(r*r + z*z,-2) *                (1);

    dxU(1,1) = -(1.0/3.0)*sin(thetab) * ( r)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (-z)
               +(1.0/3.0)*cos(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*z) * (-z)
               +(1.0/3.0)*cos(thetab) *                          pow(r*r + z*z,-2) *                (-1)
               +(1.0/6.0)*cos(thetab) * ( r)*pow(r*r + z*z,-1) * pow(r*r + z*z,-2) *                (-r)
               +(1.0/6.0)*sin(thetab) *                          (-2.0)*pow(r*r + z*z,-1)*(2.0*z) * (-r);
  }
  return dxU;
}

double AppCtx::p_exacta(Vector const& X, double t, int tag){
  double x = X(0);
  double y = X(1);
  double p = 0.0;
  //Vector nor(Vector::Zero(X.size()));
  //Vector tau(Vector::Zero(X.size()));
  if (dim == 2){
    Vector Xc(2), Y(2);
    double rb, thetab;
    double B1 = +1.0*0.5, B2 = +3.0, mu = muu(0), R = 1.0;
    Xc << XG_0[0](0), XG_0[0](1);
    Y = X - Xc;
    x = Y(0); y = Y(1);
    rb  = sqrt(x*x+y*y);
    //nor = Y/rb; tau(0) = +nor(1); tau(1) = -nor(0);
    thetab = -atan2(Y(1),Y(0))+pi/2.0;
    p = mu*(-R*R/(rb*rb*rb))*B2*(3.0*cos(thetab)*cos(thetab)-1.0);
  }

  return p;
}

Vector AppCtx::ftau_exacta(Vector const& X, Vector const& XG, Vector const& normal,
                           double t, int tag, double theta)
{
  Vector tau(dim);
  tau(0) = +normal(1); tau(1) = -normal(0);  //this tangent goes in the direction of the parametrization
  Vector f(Vector::Zero(X.size()));

  double psi = 0.0;
  psi = atan2PI(X(1)-XG(1),X(0)-XG(0));
  double B1 = +0.5, B2 = +3.0, mu = muu(0), R = 1.0;
  double uthe = (-mu/R) * (2*B1*sin(theta-psi) + 5*B2*sin(theta-psi)*cos(theta-psi));
  //f(0) = k*normal(0); f(1) = k*normal(1);
  f = uthe*tau;

  return f;
}

PetscErrorCode AppCtx::ProjOrtMatrix(Matrix3d & Qn, int it, double eps, int dim){
  Matrix3d E(Matrix3d::Zero(3,3));
  Matrix3d Id(Matrix3d::Identity(3,3));

  if (it == 0){
    if (dim == 2){
      Matrix2d Qn2(Matrix2d::Zero(2,2));
      Qn2 << Qn(0,0),Qn(0,1),Qn(1,0),Qn(1,1);
      JacobiSVD<Matrix2d> svd(Qn2, ComputeFullU | ComputeFullV);
      Matrix2d U = svd.matrixU(); Matrix2d V = svd.matrixV();
      Qn2 = U*V.transpose();
      Qn(0,0) = Qn2(0,0); Qn(0,1) = Qn2(0,1);
      Qn(1,0) = Qn2(1,0); Qn(1,1) = Qn2(1,1);
    }
    else{
      JacobiSVD<Matrix3d> svd(Qn, ComputeFullU | ComputeFullV); //svd(Qn, ComputeThinU | ComputeThinV);
      Matrix3d U = svd.matrixU(); Matrix3d V = svd.matrixV();
      Qn = U*V.transpose();
    }
  }
  else{
    for (int k = 0; k < it; k++){
      E = Id - Qn.transpose()*Qn;
      Qn = Qn + 0.5*Qn*E;
      if (E.norm() < eps)
        break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::calcSlipVelocity(Vec const& Vec_x_1_, Vec& Vec_slipv){
  PetscErrorCode      ierr(0);

  //Vec Vec_normal_aux;
  //ierr = VecCreate(PETSC_COMM_WORLD, &Vec_normal_aux);               CHKERRQ(ierr);
  //ierr = VecSetSizes(Vec_normal_aux, PETSC_DECIDE, n_dofs_v_mesh);   CHKERRQ(ierr);
  //ierr = VecSetFromOptions(Vec_normal_);                          CHKERRQ(ierr);
  //Assembly(Vec_normal_aux);
  VecZeroEntries(Vec_normal_aux);
  VecZeroEntries(Vec_rho_aux);  //to save slip_rho differences for tangent slip vel calc
  VecZeroEntries(Vec_slip_rho);
  VecZeroEntries(Vec_tangent);
  VecZeroEntries(Vec_slipv);
  cout << "Phoresis solver" << endl;
  ierr = SNESSolve(snes_s, PETSC_NULL, Vec_slip_rho);  CHKERRQ(ierr);
  //View(Vec_slip_rho, "matrizes/rhosol.m", "rs");
  //View(Vec_x_1_, "matrizes/x1.m", "x");
  getVecNormals(&Vec_x_1_, Vec_normal_aux);
  //plotFiles();

  int tag, k;
  double alphaP;
  Point const* point(NULL);//, point_i(NULL);
  const int n_nodes_total = mesh->numNodesTotal();
  VectorXi points_id = -VectorXi::Ones(20);  //more than 20? 20 maximum star(point) length
  Vector Xi(dim), Xj(dim), Nr(dim), Vs(dim), Tg(dim);
  Vector S_coefs(n_dofs_s_per_facet), bs(dim), Grad(dim), SGrad(dim);
  Matrix<double, Dynamic,1,0,20,1> Ddif(20);
  MatrixXd Xdif(20,dim), Wdif(20,20), As(dim,dim);
  VectorXi mapS(n_dofs_s_per_facet), dofs_mesh(dim);
  Tensor I(dim,dim), Pr(dim,dim);
  I.setIdentity();

if (false){// case projector by least squares
  for (int j = 0; j < n_nodes_total; j++)
  {
    point = mesh->getNodePtr(j);
    tag = point->getTag();
    if (!is_in(tag,slipvel_tags) /*&& !is_in(tag,flusoli_tags)*/)
      continue;

    mesh->connectedNodes(&*point, points_id.data());
    //cout << i << " con to " << points_id.transpose() << endl;

    point->getCoord(Xj.data(),dim);
    mapS(0) = j;
    k = 0;

    for (int i = 0; i < 20; i++){
      if (points_id[i] < 0)
        break;

      //point_i = mesh->getNodePtr(points_id[i]);
      tag = mesh->getNodePtr(points_id[i])->getTag();

      if (is_in(tag,solidonly_tags))
        continue;

      mesh->getNodePtr(points_id[i])->getCoord(Xi.data(),dim);
      Xdif.row(k) = (Xi-Xj).transpose();

      mapS(1) = points_id[i];
      VecGetValues(Vec_slip_rho, mapS.size(), mapS.data(), S_coefs.data());
      Ddif(k) = S_coefs(1) - S_coefs(0);

      Wdif(k,k) = 1./(Xi-Xj).norm();

      k++;
    }

    As = Xdif.block(0,0,k,dim).transpose()*Wdif.block(0,0,k,k)*Wdif.block(0,0,k,k)*Xdif.block(0,0,k,dim);
    bs = Xdif.block(0,0,k,dim).transpose()*Wdif.block(0,0,k,k)*Wdif.block(0,0,k,k)*Ddif.head(k);
    Grad = As.colPivHouseholderQr().solve(bs);

    //As = Wdif.block(0,0,k,k)*Wdif.block(0,0,k,k)*Xdif.block(0,0,k,2);
    //bs = Wdif.block(0,0,k,k)*Wdif.block(0,0,k,k)*Ddif.head(k)
    //Grad = As.jacobiSvd(ComputeThinU | ComputeThinV).solve(bs);
    getNodeDofs(&*point, DH_MESH, VAR_M, dofs_mesh.data());
    VecGetValues(Vec_normal_aux, dim, dofs_mesh.data(), Nr.data());

    Pr = I - Nr*Nr.transpose();
    SGrad = Pr*Grad;
    Vs = -bbG_coeff(Xj,tag)*SGrad;  //Xi means nothing here
    VecSetValues(Vec_slipv, dim, dofs_mesh.data(), Vs.data(), INSERT_VALUES);
    //cout << Xdif.jacobiSvd(ComputeThinU | ComputeThinV).solve(Ddif);
    //cout << Grad << endl << endl;
  }
}
else if(false){// first try slip vel calc
  for (int j = 0; j < n_nodes_total; j++)
  {
    point = mesh->getNodePtr(j);
    tag = point->getTag();
    if (!is_in(tag,slipvel_tags) /*&& !is_in(tag,flusoli_tags)*/)
      continue;

    mesh->connectedNodes(&*point, points_id.data());
    alphaP = 0;
    getNodeDofs(&*point, DH_MESH, VAR_M, dofs_mesh.data());
    VecGetValues(Vec_normal_aux, dim, dofs_mesh.data(), Nr.data());

    Tg(0) = -Nr(1); Tg(1) = Nr(0); if (dim == 3) {Tg(2) = Nr(2);}
    VecSetValues(Vec_tangent, dim, dofs_mesh.data(), Tg.data(), INSERT_VALUES);

    point->getCoord(Xj.data(),dim);
    mapS(0) = j;
    k = 0;

    for (int i = 0; i < 20; i++){
      if (points_id[i] < 0)
        break;

      //point_i = mesh->getNodePtr(points_id[i]);
      tag = mesh->getNodePtr(points_id[i])->getTag();

      if (!(is_in(tag,slipvel_tags)||is_in(tag,flusoli_tags)))
        continue;

      mesh->getNodePtr(points_id[i])->getCoord(Xi.data(),dim);

      alphaP = alphaP + (Xi-Xj).dot(Tg);//*(Xi-Xj).norm();
    }

    mapS(1) = j;
    VecGetValues(Vec_slip_rho, mapS.size(), mapS.data(), S_coefs.data());

    Vs = -bbG_coeff(Xj,tag)*(S_coefs(0)/2.0)*alphaP*Tg;
    VecSetValues(Vec_slipv, dim, dofs_mesh.data(), Vs.data(), INSERT_VALUES);
  }
}
else if(true){// discretization of surface gradient over the edges
  VectorXi          facet_nodes(nodes_per_facet);
  MatrixXd          x_coefs(nodes_per_facet, dim);                // coordenadas nodais da célula
  MatrixXd          x_coefs_trans(dim, nodes_per_facet);
  Vector            X(dim);
  Vector            tangent_f(dim); //normal_f(dim);
  Tensor            F(dim,dim-1);
  VectorXi          map(n_dofs_u_per_facet);
  int               tag, sign_;
  bool              is_fsi, is_slv;//is_surface, is_solid, is_cl;
  double            betashi;

  facet_iterator facet = mesh->facetBegin();
  facet_iterator facet_end = mesh->facetEnd();
  for (; facet != facet_end; ++facet)
  {
    tag = facet->getTag();

    //is_surface = is_in(tag, interface_tags);
    //is_solid   = is_in(tag, solid_tags);
    //is_cl      = is_in(tag, triple_tags);
    is_fsi     = is_in(tag, flusoli_tags);
    is_slv     = is_in(tag, slipvel_tags);

    if ( !(is_fsi || is_slv) )// not using is_fsi disconsider the both edges contribution to the tangent
      continue;

    dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(map.data(), &*facet);
    mesh->getFacetNodesId(&*facet, facet_nodes.data());
    VecGetValues(Vec_x_1_, map.size(), map.data(), x_coefs.data());
    //mesh->getNodesCoords(facet_nodes.begin(), facet_nodes.end(), x_coefs.data());
    x_coefs_trans = x_coefs.transpose();

    sign_ = 1;
    cell_handler cc = mesh->getCell(facet->getIncidCell());
    cell_handler oc = mesh->getCell(cc->getIncidCell(facet->getPosition()));
    if ( oc.isValid()  )
      if ( oc->getTag() > cc->getTag() )
        sign_ = -1;

//    normal_f(0) = x_coefs_trans(1,1)-x_coefs_trans(1,0);
//    normal_f(1) = x_coefs_trans(0,0)-x_coefs_trans(0,1);
//    normal_f *= sign_;
    // mass conserving tangent
    tangent_f(0) = x_coefs_trans(0,1)-x_coefs_trans(0,0);
    tangent_f(1) = x_coefs_trans(1,1)-x_coefs_trans(1,0);
    tangent_f.normalize();
    tangent_f *= ((x_coefs_trans.col(0)-x_coefs_trans.col(1)).norm());
    tangent_f *= sign_;
      //contribution to node a
    VecSetValues(Vec_tangent, dim, map.data()+0*dim, tangent_f.data(), ADD_VALUES); //map.data(): pointer to position 0,
      //contribution to node b                                        //map.data()+1*dim: pointer to position
    VecSetValues(Vec_tangent, dim, map.data()+1*dim, tangent_f.data(), ADD_VALUES); //dim, so takes 0 and 1 first, then 2 and 3
    // rho differences
    X = (x_coefs_trans.col(0)+x_coefs_trans.col(1))/2;   //evaluate bbG_coeff at the midpoint of the edge
    dof_handler[DH_SLIP].getVariable(VAR_S).getFacetDofs(mapS.data(), &*facet);
    VecGetValues(Vec_slip_rho, mapS.size(), mapS.data(), S_coefs.data());
    betashi = (S_coefs(1) - S_coefs(0))*(-bbG_coeff(X, tag));
    betashi *= sign_;
      //contribution to node a
    VecSetValues(Vec_rho_aux, 1, mapS.data()+0, &betashi, ADD_VALUES); //map.data(): pointer to position 0,
      //contribution to node b                                        //map.data()+1*dim: pointer to position
    VecSetValues(Vec_rho_aux, 1, mapS.data()+1, &betashi, ADD_VALUES); //dim, so takes 0 and 1 first, then 2 and 3
  }  // end for facet


  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    tag = point->getTag();

    //is_surface = is_in(tag, interface_tags);
    //is_solid   = is_in(tag, solid_tags);
    //is_cl      = is_in(tag, triple_tags);
    is_fsi     = is_in(tag, flusoli_tags);
    is_slv     = is_in(tag, slipvel_tags);

    if ( !(/*is_surface || is_solid || is_cl || mesh->inBoundary(&*point) || is_fsi || */is_slv) )
      continue;

    getNodeDofs(&*point, DH_MESH, VAR_M, map.data());
    VecGetValues(Vec_tangent, dim, map.data(),tangent_f.data());
    betashi = 1/tangent_f.norm();
    tangent_f.normalize();
    VecSetValues(Vec_tangent, dim, map.data(),tangent_f.data(), INSERT_VALUES);

    getNodeDofs(&*point, DH_SLIP, VAR_S, mapS.data());
    VecGetValues(Vec_rho_aux, 1, mapS.data(), S_coefs.data());
    betashi = betashi*S_coefs(0);
    VecSetValues(Vec_rho_aux, 1, mapS.data()+0, &betashi, INSERT_VALUES); //map.data(): pointer to position 0,

    Vs = betashi*tangent_f;
    VecSetValues(Vec_slipv, dim, map.data(), Vs.data(), INSERT_VALUES);

  }
}
  //View(Vec_normal_aux,"matrizes/nr1.m","n1"); View(Vec_x_1,"matrizes/xv1.m","x1");
  //View(Vec_tangent,"matrizes/tv1.m","t1"); View(Vec_slipv,"matrizes/sv1.m","s1"); View(Vec_slip_rho,"matrizes/sr1.m","r1");
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::timeAdapt(){

  VecCopy(Vec_ups_0,Vec_ups_m1);  //t^{n-2} is now at the position of the old t^{n-1}
                                  //Vec_ups_m1 is now t^{n-1} because the half time adaptation
  VecZeroEntries(Vec_ups_time_aux);
  VecAXPY(Vec_ups_time_aux,0.5,Vec_ups_1);  //time adaptation to the half time step
  VecAXPY(Vec_ups_time_aux,0.5,Vec_ups_0);  //" " " "
  VecCopy(Vec_ups_time_aux,Vec_ups_0);      //Vec_ups_0 is now t^{n-1/2}

  VecZeroEntries(Vec_x_time_aux);
  VecAXPY(Vec_x_time_aux,0.5,Vec_x_1);  //time adaptation to the half time step
  VecAXPY(Vec_x_time_aux,0.5,Vec_x_0);  //" " " "
  VecCopy(Vec_x_time_aux,Vec_x_0);      //Vec_ups_0 is now t^{n-1/2}
  VecZeroEntries(Vec_x_time_aux);

//  if (is_bdf3){

//  }

  calcSlipVelocity(Vec_x_1, Vec_slipv_1);


  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::saveDOFSinfo(int step){
  VectorXd v_coeffs_s(LZ*n_solids);
  VectorXi mapvs(LZ*n_solids);
  Vector3d Xgg;
  Matrix3d Qrt;
  double theta, modes;

  // Printing mech. dofs information //////////////////////////////////////////////////
  if (fprint_hgv && is_sfip){
    if ((time_step%print_step)==0 || time_step == (maxts-1)){
      getSolidVolume();
      for (int S = 0; S < LZ*n_solids; S++){
        mapvs[S] = n_unknowns_u + n_unknowns_p + S;
      }
      VecGetValues(Vec_ups_1,mapvs.size(),mapvs.data(),v_coeffs_s.data());
      filg.open(grac,iostream::app);
      filg.precision(15);
      filg.setf(iostream::fixed, iostream::floatfield);
      filv.open(velc,iostream::app);
      filv.precision(15);
      filv.setf(iostream::fixed, iostream::floatfield);
      filt.open(rotc,iostream::app);
      filt.precision(15);
      filt.setf(iostream::fixed, iostream::floatfield);
      filg << current_time << " ";
      filv << current_time << " ";
      filt << current_time << " ";
      for (int S = 0; S < n_solids/*(n_solids-1)*/; S++){
        if (step == 0){Xgg = XG_0[S]; theta = theta_0[S]; Qrt = Q_0[S]; modes = modes_0[S];}
        else          {Xgg = XG_1[S]; theta = theta_1[S]; Qrt = Q_1[S]; modes = modes_1[S];}
        filg << Xgg(0) << " " << Xgg(1); if (dim == 3){filg << " " << Xgg(2);} filg << " " << theta; if (is_sfim){filg << " " << modes;}
        if (S < (n_solids-1)){filg << " ";}// << VV[S] << " ";
        filv << v_coeffs_s(LZ*S) << " " << v_coeffs_s(LZ*S+1)  << " " << v_coeffs_s(LZ*S+2); if (is_sfim){filv << " " << v_coeffs_s(LZ*S+3);}
        if (S < (n_solids-1)){filv << " ";}
        filt << Qrt(0,0) << " " << Qrt(0,1) << " " << Qrt(0,2) << " "
             << Qrt(1,0) << " " << Qrt(1,1) << " " << Qrt(1,2) << " "
             << Qrt(2,0) << " " << Qrt(2,1) << " " << Qrt(2,2) << " " << VV[n_solids-1] ; if (S < (n_solids-1)){filt << " ";}
      }
      //if (step == 0){Xgg = XG_0[n_solids-1]; theta = theta_0[n_solids-1];} else{Xgg = XG_1[n_solids-1]; theta = theta_1[n_solids-1];}
      //filg << Xgg(0) << " " << Xgg(1); if (dim == 3){filg << " " << Xgg(2);} filg << " " << theta;//" " << VV[n_solids-1] << endl;
      //filv << v_coeffs_s(LZ*(n_solids-1)) << " " << v_coeffs_s(LZ*(n_solids-1)+1) << " "
      //     << v_coeffs_s(LZ*(n_solids-1)+2);
      if (dim == 3){
        filv << " " << v_coeffs_s(LZ*(n_solids-1)+3) << " " << v_coeffs_s(LZ*(n_solids-1)+4) << " "
            << v_coeffs_s(LZ*(n_solids-1)+5);
      }
      filg << endl; filv << endl; filt << endl;
      filg.close(); filv.close(); filt.close();
    }
  } // end Printing mech. dofs information //////////////////////////////////////////////////

  //Testing the angular advancing//
  Matrix3d QrfT = RotM(theta_1[0],Q_1[0],1);
  cout << "Determinant of Q^{n} = " << Q_1[0].determinant() << endl;
  cout << "Matrix Difference =\n" << Q_1[0] - QrfT << endl << endl;
  cout << "Norm of difference = " << (Q_1[0] - QrfT).norm() << endl;
  //////////////////////////////////////////////////////////

 PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::saveDOFSinfo_Re_Vel(){
  VectorXd v_coeffs_s(LZ*n_solids);
  VectorXi mapvs(LZ*n_solids);
  Vector3d Xgg;

  // Printing mech. dofs information //////////////////////////////////////////////////
  if (fprint_hgv && is_sfip){
    if ((time_step%print_step)==0 || time_step == (maxts-1)){
      getSolidVolume();
      for (int S = 0; S < LZ*n_solids; S++){
        mapvs[S] = n_unknowns_u + n_unknowns_p + S;
      }
      VecGetValues(Vec_ups_1,mapvs.size(),mapvs.data(),v_coeffs_s.data());
      filg.open(grac,iostream::app);
      filg.precision(15);
      filg.setf(iostream::fixed, iostream::floatfield);
      filv.open(velc,iostream::app);
      filv.precision(15);
      filv.setf(iostream::fixed, iostream::floatfield);
      //double alp = 100.0, bet = 0.018, kst = (double)time_step, alp1 = 8.0;
      //filg << pow(10.0,(kst+000.0)/alp) << " ";
      //filv << pow(10.0,(kst+000.0)/alp) << " ";
      //filg << pow(10.0,2.0 + bet*kst + (1.0/(alp*alp) - bet/alp)*kst*kst) << " ";
      //filv << pow(10.0,2.0 + bet*kst + (1.0/(alp*alp) - bet/alp)*kst*kst) << " ";
/*      if (kst <= alp1){
        filg << pow(10.0,(kst+alp1)/alp1) << " ";
        filv << pow(10.0,(kst+alp1)/alp1) << " ";
      }
      else{
        filg << pow(10.0,2.0 + bet*(kst-alp1) + (1.0/(alp*alp) - bet/alp)*(kst-alp1)*(kst-alp1)) << " ";
        filv << pow(10.0,2.0 + bet*(kst-alp1) + (1.0/(alp*alp) - bet/alp)*(kst-alp1)*(kst-alp1)) << " ";
      }*/

      double kst = (double)time_step, p1 = -3.0, alp1 = 10.0, alp2 = 20.0, alp3 = 30.0, alp4 = 40.0, alp5 = 50.0;
      if (kst <= alp1){
        filg <<  pow(10.0,p1  ) + (kst     )*(pow(10.0,p1+1) - pow(10.0,p1  ))/(alp1     ) << " ";
        filv <<  pow(10.0,p1  ) + (kst     )*(pow(10.0,p1+1) - pow(10.0,p1  ))/(alp1     ) << " ";
      }
      else if (kst > alp1 && kst <= alp2){
        filg <<  pow(10.0,p1+1) + (kst-alp1)*(pow(10.0,p1+2) - pow(10.0,p1+1))/(alp2-alp1) << " ";
        filv <<  pow(10.0,p1+1) + (kst-alp1)*(pow(10.0,p1+2) - pow(10.0,p1+1))/(alp2-alp1) << " ";
      }
      else if (kst > alp2 && kst <= alp3){
        filg <<  pow(10.0,p1+2) + (kst-alp2)*(pow(10.0,p1+3) - pow(10.0,p1+2))/(alp3-alp2) << " ";
        filv <<  pow(10.0,p1+2) + (kst-alp2)*(pow(10.0,p1+3) - pow(10.0,p1+2))/(alp3-alp2) << " ";
      }
      else if (kst > alp3 && kst <= alp4){
        filg <<  pow(10.0,p1+3) + (kst-alp3)*(pow(10.0,p1+4) - pow(10.0,p1+3))/(alp4-alp3) << " ";
        filv <<  pow(10.0,p1+3) + (kst-alp3)*(pow(10.0,p1+4) - pow(10.0,p1+3))/(alp4-alp3) << " ";
      }
      else if (kst > alp4 && kst <= alp5){
        filg <<  pow(10.0,p1+4) + (kst-alp4)*(pow(10.0,p1+5) - pow(10.0,p1+4))/(alp5-alp4) << " ";
        filv <<  pow(10.0,p1+4) + (kst-alp4)*(pow(10.0,p1+5) - pow(10.0,p1+4))/(alp5-alp4) << " ";
      }
      else{
        filg <<  -1000000 << " ";
        filv <<  -1000000 << " ";
      }


      for (int S = 0; S < (n_solids-1); S++){
        Xgg = XG_0[S];
        filg << Xgg(0) << " " << Xgg(1); if (dim == 3){filg << " " << Xgg(2);} filg << " " << theta_0[S] << " ";// << VV[S] << " ";
        filv << v_coeffs_s(LZ*S) << " " << v_coeffs_s(LZ*S+1) << " " << v_coeffs_s(LZ*S+2) << " ";
      }
      Xgg = XG_0[n_solids-1];
      filg << Xgg(0) << " " << Xgg(1); if (dim == 3){filg << " " << Xgg(2);} filg << " " << theta_0[n_solids-1] << endl;//" " << VV[n_solids-1] << endl;
      filv << v_coeffs_s(LZ*(n_solids-1)) << " " << v_coeffs_s(LZ*(n_solids-1)+1) << " "
           << v_coeffs_s(LZ*(n_solids-1)+2);
      if (dim == 3){
        filv << " " << v_coeffs_s(LZ*(n_solids-1)+3) << " " << v_coeffs_s(LZ*(n_solids-1)+4) << " "
            << v_coeffs_s(LZ*(n_solids-1)+5);
      }
      filv << endl;
      filg.close();  //TT++;
      filv.close();
    }
  } // end Printing mech. dofs information //////////////////////////////////////////////////

 PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::extractForce(bool print){
  int tag, nod_id, nod_vs, pID;
  double theta;
  Point const* point(NULL);//, point_i(NULL);
  const int n_nodes_total = mesh->numNodesTotal();
  Vector  Xj(dim), Fd(dim), Nr(dim), Tg(dim), Sv(dim), Ft(dim), Ftau(dim);
  Vector  Zf(Vector::Zero(LZ)), Uf(dim), U(dim);
  VectorXi dofs_mesh(dim), dofs_U(dim), dofs_fs(LZ);
  Vector3d XG;

  ofstream filfd, filfd_info;
  char fdc[PETSC_MAX_PATH_LEN], fdc_info[PETSC_MAX_PATH_LEN];

  if (print){
    sprintf(fdc,"%s/theta_Fd_file_%06d.txt",filehist_out.c_str(),time_step);
    sprintf(fdc_info,"%s/theta_Fd_file_info.txt",filehist_out.c_str());
    filfd.open(fdc);
    filfd.precision(15);

    filfd_info.open(fdc_info);
    filfd_info << "Point ID   Theta   Force Fd(dim)   Force Ft(dim)   Slip Vel(dim)   Normal(dim)   Tangent(dim)   PointCoords(dim)";
    filfd_info.close();
  }

  for (int j = 0; j < n_nodes_total; j++){
    point = mesh->getNodePtr(j);
    tag = point->getTag();
    nod_id = is_in_id(tag,flusoli_tags);
    nod_vs = is_in_id(tag,slipvel_tags);
    if (!nod_vs && !nod_id)
      continue;

    pID = mesh->getPointId(&*point);
    point->getCoord(Xj.data(),dim);
    getNodeDofs(&*point, DH_MESH, VAR_M, dofs_mesh.data());
    VecGetValues(Vec_fdis_0, dim, dofs_mesh.data(), Fd.data());
    VecGetValues(Vec_slipv_0, dim, dofs_mesh.data(), Sv.data());
    VecGetValues(Vec_ftau_0, dim, dofs_mesh.data(), Ft.data());
    VecGetValues(Vec_normal, dim, dofs_mesh.data(), Nr.data());
    VecGetValues(Vec_tangent, dim, dofs_mesh.data(), Tg.data());

    XG = XG_0[nod_vs+nod_id-1];
    theta = atan2PI(Xj(1)-XG(1),Xj(0)-XG(0));

    if (nod_vs){
      VecGetValues(Vec_slipv_0, dim, dofs_mesh.data(), Sv.data());
      Ft(0) = Fd.dot(Nr); Ft(1) = Fd.dot(Tg);
      Ftau = Fd.dot(Tg)*Tg;
      VecSetValues(Vec_ftau_0, dim, dofs_mesh.data(), Ftau.data(), INSERT_VALUES);
      Ft = Ftau;
    }
    else if (nod_id){
      for (int l = 0; l < LZ; l++){
        dofs_fs(l) = n_unknowns_u + n_unknowns_p + LZ*(nod_id-1) + l;
      }
      VecGetValues(Vec_ups_1, LZ, dofs_fs.data(), Zf.data());
      //Uf = SolidVel(Xj, XG, Zf, dim);
      Uf = SolidVelGen(Xj, XG, theta_0[nod_vs+nod_id-1], modes_0[nod_vs+nod_id-1], Zf, dim, is_axis);
      getNodeDofs(&*point,DH_UNKM,VAR_U,dofs_U.data());
      VecGetValues(Vec_ups_1, dim, dofs_U.data(), U.data());
      Sv = U - Uf;  //cout << VS.transpose() << endl;
      VecSetValues(Vec_slipv_0, dim, dofs_mesh.data(), Sv.data(), INSERT_VALUES);// before Vec_slipv_1
      VecSetValues(Vec_slipv_1, dim, dofs_mesh.data(), Sv.data(), INSERT_VALUES);// before Vec_slipv_1
      if (is_unksv){
        Ft = FtauForce(Xj, XG, Nr, dim, tag, theta_0[nod_vs+nod_id-1], Kforp, nforp, current_time,
                       Sv, Q_0[nod_vs+nod_id-1], thetaDOF, Kcte);
        VecSetValues(Vec_ftau_0, dim, dofs_mesh.data(), Ft.data(), INSERT_VALUES);// before Vec_slipv_1
      }
    }
    if (print){
      if (j == n_nodes_total-1)
        filfd << pID << " " << theta << " " << Fd(0) << " " << Fd(1) << " "
                                            << Ft(0) << " " << Ft(1) << " "
                                            << Sv(0) << " " << Sv(1) << " "
                                            << Nr(0) << " " << Nr(1) << " "
                                            << Tg(0) << " " << Tg(1) << " "
                                            << Xj(0) << " " << Xj(1);
      else
        filfd << pID << " " << theta << " " << Fd(0) << " " << Fd(1) << " "
                                            << Ft(0) << " " << Ft(1) << " "
                                            << Sv(0) << " " << Sv(1) << " "
                                            << Nr(0) << " " << Nr(1) << " "
                                            << Tg(0) << " " << Tg(1) << " "
                                            << Xj(0) << " " << Xj(1) << endl;
    }
  }
  if (print)
    filfd.close();

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::importSurfaceInfo(int time_step){

  char fdc[PETSC_MAX_PATH_LEN];
  sprintf(fdc,"%s%06d.txt",filesvfd.c_str(),time_step);
  FD_file.clear(); FT_file.clear(); SV_file.clear(); NR_file.clear(); TG_file.clear(); pIDs.clear();

  int pID = 0;
  int N = 0;
  long double theta = 0.0, dID;
  long double fd1 = 0.0, fd2 = 0.0, fd3 = 0.0, sv1 = 0.0, sv2 = 0.0, sv3 = 0.0,
         nr1 = 0.0, nr2 = 0.0, nr3 = 0.0, tg1 = 0.0, tg2 = 0.0, tg3 = 0.0,
         ft1 = 0.0, ft2 = 0.0, ft3 = 0.0, x = 0.0, y = 0.0, z = 0.0;
  Vector3d FD, FT, SV, NR, TG;
  ifstream is;
  is.open(fdc,ifstream::in);
  if (!is.good()) {cout << "svfd file not found" << endl; throw;}
  while(!is.eof()){
    is >> dID; is >> theta; pID = (int) dID;
    is >> fd1; is >> fd2; if(dim == 3){is >> fd3;}
    is >> ft1; is >> ft2; if(dim == 3){is >> ft3;}
    is >> sv1; is >> sv2; if(dim == 3){is >> sv3;}
    is >> nr1; is >> nr2; if(dim == 3){is >> nr3;}
    is >> tg1; is >> tg2; if(dim == 3){is >> tg3;}
    is >> x;   is >> y;   if(dim == 3){is >> z;}
    FD << fd1, fd2, fd3; FT << ft1, ft2, ft3; SV << sv1, sv2, sv3;
    NR << nr1, nr2, nr3; TG << tg1, tg2, tg3;
    FD_file.push_back(FD); FT_file.push_back(FT); SV_file.push_back(SV);
    NR_file.push_back(NR); TG_file.push_back(TG);
    pIDs.push_back(pID);
    N++;
    //cout << pID << " " << theta << " " << fd1 << " " << fd2 << " " << ft1 << " " << ft2
    //    << " " << sv1 << " " << sv2 << " " << nr1 << " " << nr2
    //    << " " << tg1 << " " << tg2 << " " << x << " " << y << endl;
  }
  is.close();
  FD_file.resize(N); FT_file.resize(N); SV_file.resize(N); NR_file.resize(N); TG_file.resize(N);
  pIDs.resize(N);

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::SarclTest(){
  int tag, nod_id, nod_vs, pID;
  Point const* point(NULL);//, point_i(NULL);
  const int n_nodes_total = mesh->numNodesTotal();
  Vector  Xj(dim);
  Vector3d XG;

  char vspc[PETSC_MAX_PATH_LEN];
  sprintf(vspc,"%s/HistPar.txt",filehist_out.c_str());
  ofstream filp;
  filp.open(vspc); filp.close();
  filp.open(vspc,iostream::app);
  filp.precision(15);

  sprintf(vspc,"%s/HistVsp.txt",filehist_out.c_str());
  ofstream fils;
  fils.open(vspc); fils.close();
  fils.open(vspc,iostream::app);
  fils.precision(15);

  for (int j = 0; j < n_nodes_total; j++){
    point = mesh->getNodePtr(j);
    tag = point->getTag();
    nod_id = is_in_id(tag,flusoli_tags);
    nod_vs = is_in_id(tag,slipvel_tags);
    if (!nod_vs && !nod_id)
      continue;

    pID = mesh->getPointId(&*point);
    point->getCoord(Xj.data(),dim);
    XG = XG_0[nod_vs+nod_id-1];
    //double S = S_arcl(Xj(1), XG(1));
        //cout << "For point " << pID << " coord " << Xj(0) << " " << Xj(1) <<
        //        " centre " << XG(0) << " " << XG(1) << " arc length " << S << endl;

    Matrix3d Qr(Matrix3d::Zero(3,3));
    Vector3d Xref(Vector3d::Zero(3)), X3(Vector::Zero(3));
    X3(0) = Xj(0); X3(1) = Xj(1);
    Xref = RotM(theta_0[nod_vs+nod_id-1],Qr,thetaDOF).transpose()*(X3 - XG);
    double a = 110.0e-0;
    double L = S_arcl(-a,0.0);  //cout << "for " << Xref(0) << "  " << W << "  " << L << endl;
    double W = S_arcl(Xref(0), 0.0), Wr = W;
    if (Xref(1) < 0){Wr = 2*L - W;}
    filp << pID << " " << Xref(0) << " " << Wr << " ";
    fils << pID << " " << Xref(0) << " " << Wr << " ";

    for (int tsp = 0; tsp < PI+1; tsp++){
      double t = dt*tsp;
      double uthe = 0.0;
      double a = 110.0e-0, Tp = 0.2, omega = -2*pi/Tp, eta = 4;
      double S = 0.0;
      double L = S_arcl(-a,0.0);  //cout << "for " << X(1) << "  " << S << "  " << L << endl;
      double K = Kforp*L;  //0.015*L;
      double n = (double)nforp /*9*/, k = 2*pi/L * n;
      //uthe = tanh(eta*sin(pi*S/L)) * A*omega*sin(k*S - omega*t);

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
      filp << " " << S; fils << " " << uthe;
    }
    filp << endl; fils << endl;
  }
  filp.close(); fils.close();
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::CheckInvertedElements(){
  MatrixXd            x_coefs_c_mid_trans(dim, nodes_per_cell); // n+utheta
  MatrixXd            x_coefs_c_new(nodes_per_cell, dim);       // n+1
  MatrixXd            x_coefs_c_new_trans(dim, nodes_per_cell); // n+1
  MatrixXd            x_coefs_c_old(nodes_per_cell, dim);       // n
  MatrixXd            x_coefs_c_old_trans(dim, nodes_per_cell); // n
  VectorXi            cell_nodes(nodes_per_cell);
  VectorXi            mapM_c(dim*nodes_per_cell);
  Tensor              F_c_mid(dim,dim);       // n+utheta
  Tensor              invF_c_mid(dim,dim);    // n+utheta
  double              J_mid;
  inverted_elem = PETSC_FALSE;

  ////////////////////////////////////////////////// STARTING CELL ITERATION //////////////////////////////////////////////////
  //const int tid = omp_get_thread_num();
  //const int nthreads = omp_get_num_threads();
  cell_iterator cell = mesh->cellBegin();//(tid,nthreads);    //cell_iterator cell = mesh->cellBegin();
  cell_iterator cell_end = mesh->cellEnd();//(tid,nthreads);  //cell_iterator cell_end = mesh->cellEnd();
  for (; cell != cell_end; ++cell)
  {
    mesh->getCellNodesId(&*cell, cell_nodes.data());  //cout << cell_nodes.transpose() << endl;
    dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);  //cout << mapM_c.transpose() << endl;  //unk. global ID's

    VecGetValues(Vec_x_0, mapM_c.size(), mapM_c.data(), x_coefs_c_old.data());  //cout << x_coefs_c_old << endl << endl;
    VecGetValues(Vec_x_1, mapM_c.size(), mapM_c.data(), x_coefs_c_new.data());  //cout << x_coefs_c_new << endl << endl;

    x_coefs_c_old_trans = x_coefs_c_old.transpose();
    x_coefs_c_new_trans = x_coefs_c_new.transpose();
    x_coefs_c_mid_trans = vtheta*x_coefs_c_new_trans + (1.-vtheta)*x_coefs_c_old_trans;

    ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
    for (int qp = 0; qp < n_qpts_cell; ++qp)
    {
      //F_c_mid = Tensor::Zero(dim,dim);  //Zero(dim,dim);
      F_c_mid = x_coefs_c_mid_trans * dLqsi_c[qp];  // (dim x nodes_per_cell) (nodes_per_cell x dim)
      inverseAndDet(F_c_mid,dim,invF_c_mid,J_mid);

      //inverted and degenerated element test//////////////////////////////////////////////////
      if (J_mid < 1.e-20)
      {
        printf("in formCellFunction:\n");
        std::cout << "erro: jacobiana da integral não invertível: ";
        std::cout << "J_mid = " << J_mid << endl;
        cout << "trans(f) matrix:\n" << F_c_mid << endl;
        cout << "x coefs mid:" << endl;
        cout << x_coefs_c_mid_trans.transpose() << endl;
        cout << "-----" << endl;
        cout << "cell id: " << mesh->getCellId(&*cell) << endl;
        cout << "cell Contig id: " << mesh->getCellContigId( mesh->getCellId(&*cell) ) << endl;
        cout << "cell nodes:\n" << cell_nodes.transpose() << endl;
        cout << "mapM :\n" << mapM_c.transpose() << endl;
        inverted_elem = PETSC_TRUE;
        freePetscObjs();
        PetscFinalize();
        cout << "Petsc Objects Destroyed. Saving Inverted Mesh." << endl;
        throw;
        //PetscFunctionReturn(0);
      }
    }//end for qp
  }//end for cell

  PetscFunctionReturn(0);
}

Vector AppCtx::BFields_from_file(int pID, int opt)
{
  Vector VR(Vector::Zero(dim));
  int point = std::find(pIDs.begin(),pIDs.end(),pID) - pIDs.begin();
  if (opt == 0){
    Vector3d FD = FD_file[point];
    VR(0) = FD(0); VR(1) = FD(1); if(dim == 3){VR(2) = FD(2);}
  }
  else if (opt == 1){
    Vector3d FT = FT_file[point];
    VR(0) = FT(0); VR(1) = FT(1); if(dim == 3){VR(2) = FT(2);}
  }
  else if (opt == 2){
    Vector3d SV = SV_file[point];
    VR(0) = SV(0); VR(1) = SV(1); if(dim == 3){VR(2) = SV(2);}
  }
  else if (opt == 3){
    Vector3d NR = NR_file[point];
    VR(0) = NR(0); VR(1) = NR(1); if(dim == 3){VR(2) = NR(2);}
  }
  else if (opt == 4){
    Vector3d TG = TG_file[point];
    VR(0) = TG(0); VR(1) = TG(1); if(dim == 3){VR(2) = TG(2);}
  }
  return VR;
}

void AppCtx::printProblemInfo(){
  cout << endl; cout << "mesh: " << filename << endl;
  if (is_curvt) {cout << "\n2D curved triangle\n";}
  mesh->printInfo();
  cout << "\n# velocity unknowns: " << n_unknowns_u;
  cout << "\n# pressure unknowns: " << n_unknowns_p << " = "
                                    << mesh->numVertices() << " + " << n_unknowns_p-mesh->numVertices();
//  cout << "\n# total unknowns: " << n_unknowns << end;
  if (n_solids){
    cout << "\n# velocity fluid only unknowns: " << n_unknowns_u - dim*(n_nodes_so+n_nodes_fsi+n_nodes_sv);
    cout << "\n# velocity solid only unknowns: " << n_nodes_so*dim;
    cout << "\n# velocity solid intf unknowns: " << n_unknowns_z << " (" <<
                                                   (n_nodes_fsi+n_nodes_sv) << ") = " <<
                                                   n_unknowns_z*(n_nodes_fsi+n_nodes_sv);
    cout << "\n# total unknowns: " << n_unknowns_fs << " = " << n_unknowns_fs - dim*(n_nodes_fsi+n_nodes_sv)
                                                         << " + " << dim*(n_nodes_fsi+n_nodes_sv);
    cout << "\n# solids: " << n_solids;
    cout << "\n# surface facets: " << n_facets_fsi << endl;
    cout << "  unknowns distribution: " << 0 << "-" << n_unknowns_u-1 <<
            ", " << n_unknowns_u << "-" << n_unknowns_u + n_unknowns_p - 1 <<
            ", " << n_unknowns_u + n_unknowns_p << "-"  << n_unknowns_u + n_unknowns_p +
                                                                     n_unknowns_z - 1;
    //if (is_unksv){
    //cout << ", " << n_unknowns_ups << "-"  << n_unknowns_ups + n_unknowns_u - 1;
    //}
  }
  cout << endl;
  mesh->printStatistics();
  mesh->timer.printTimes();
  cout << endl;
  if (n_solids){
    cout << "flusoli_tags (body Neumann tags) = ";
    if ((int)flusoli_tags.size() > 0){cout << flusoli_tags[0] << " - " << flusoli_tags[(int)flusoli_tags.size()-1] << endl;}
    else {cout << "No tags" << endl;}
    cout << "slipvel_tags (body Dirichlet tags) = ";
    if ((int)slipvel_tags.size() > 0){cout << slipvel_tags[0] << " - " << slipvel_tags[(int)slipvel_tags.size()-1] << endl;}
    else {cout << "No tags" << endl;}
    cout << endl;
  }
}

void AppCtx::freePetscObjs()
{
  printf("\nDestroying PETSC objects... ");

  Destroy(Mat_Jac_fs); //Destroy(Mat_Jac);
  Destroy(Mat_Jac_m);

  // No Interpolated Vectors in Mesh Adaptation /////////////////////////
  Destroy(Vec_res_fs); //Destroy(Vec_res);
  Destroy(Vec_res_m);
  Destroy(Vec_v_mid);
  Destroy(Vec_v_1);
  Destroy(Vec_normal);
  Destroy(Vec_tangent);
  if (is_bdf2 || is_bdf3){Destroy(Vec_x_cur);}
  if (time_adapt){
    Destroy(Vec_ups_time_aux);
    Destroy(Vec_x_time_aux);
  }

  // Interpolated Vectors in Mesh Adaptation /////////////////////////
  Destroy(Vec_ups_0); //Destroy(Vec_up_0);
  Destroy(Vec_ups_1); //Destroy(Vec_up_1);
  Destroy(Vec_x_0);
  Destroy(Vec_x_1);
  Destroy(Vec_ups_m1);
  Destroy(Vec_x_aux);
  if (is_bdf3)
  {
    Destroy(Vec_ups_m2);
  }
  if (is_slipv){
    Destroy(Vec_slipv_0);
    Destroy(Vec_slipv_1);
    Destroy(Vec_slipv_m1);
    if (is_bdf3){Destroy(Vec_slipv_m2);}
  }

  // pestsc contexts //////////////////////////////////////////////////
  //Destroy(ksp); //Destroy(snes); //SNESDestroy(&snes);
  SNESDestroy(&snes_m);  //snes destroys ksp, pc and linearsearch
  SNESDestroy(&snes_fs);

  // Swimmer //////////////////////////////////////////////////
  if (is_sslv){
    Destroy(Vec_res_s);
    Destroy(Vec_slip_rho);
    Destroy(Vec_normal_aux);
    Destroy(Vec_rho_aux);
    Destroy(Mat_Jac_s);
    SNESDestroy(&snes_s);
  }

  // Dissipative Force //////////////////////////////////////////////////
  if (is_sfip){
    Destroy(Vec_res_Fdis);
    Destroy(Vec_fdis_0);
    Destroy(Vec_ftau_0);
    Destroy(Mat_Jac_fd);
    SNESDestroy(&snes_fd);
  }

  printf(" done.\n");
}

void GetDataVelocity::get_vec(int id, Real * vec_out) const
{
  Point const* point = user.mesh->getNodePtr(id);
  std::vector<int> dofs(user.dim);

  user.getNodeDofs(&*point, DH_UNKM, VAR_U, dofs.data());
  for (int i = 0; i < user.dim; ++i)
    vec_out[i] = q_array[*(dofs.data()+i)];
 /*
  int tag = point->getTag();
  int nod_id = is_in_id(tag,user.flusoli_tags);
  int nod_is = is_in_id(tag,user.solidonly_tags);
  Vector Xp(user.dim), Z(user.LZ), U(user.dim);
  Vector3d Xg;

  if (nod_id || nod_is){
    Xg = user.XG_1[(nod_id+nod_is)-1];
    point->getCoord(Xp.data(),user.dim);
    for (int l = 0; l < user.LZ; l++){
      Z(l) = q_array[user.n_unknowns_u + user.n_unknowns_p + user.LZ*(nod_id+nod_is) - user.LZ + l];
    }
    U = SolidVel(Xp,Xg,Z,user.dim);
    for (int i = 0; i < user.dim; ++i)
      vec_out[i] = U(i);
  }
  else{
    user.getNodeDofs(&*point, DH_UNKM, VAR_U, dofs.data());
    for (int i = 0; i < user.dim; ++i)
      vec_out[i] = q_array[*(dofs.data()+i)];
  }
*/
}

double GetDataPressure::get_data_r(int nodeid) const
{
  Point const*const point = user.mesh->getNodePtr(nodeid);
  if (!user.mesh->isVertex(point))
  {
    int dofs[3];
    // position of the node at edge
    const int m = point->getPosition() - user.mesh->numVerticesPerCell();
    Cell const*const cell = user.mesh->getCellPtr(point->getIncidCell());
    if (user.dim==3)
    {
      const int edge_id = cell->getCornerId(m);
      user.dof_handler[DH_UNKM].getVariable(VAR_P).getCornerDofs(dofs, user.mesh->getCornerPtr(edge_id));
    }
    else
    {
      const int edge_id = cell->getFacetId(m);
      user.dof_handler[DH_UNKM].getVariable(VAR_P).getFacetDofs(dofs, user.mesh->getFacetPtr(edge_id));
    }
    return (q_array[dofs[0]] + q_array[dofs[1]])/2.;
  }
  int dof;
  user.dof_handler[DH_UNKM].getVariable(VAR_P).getVertexDofs(&dof, point);
  return q_array[dof];
}

double GetDataPressCellVersion::get_data_r(int cellid) const
{
  // assume que só há 1 grau de liberdade na célula
  int dof[user.dof_handler[DH_UNKM].getVariable(VAR_P).numDofsPerCell()];
  user.dof_handler[DH_UNKM].getVariable(VAR_P).getCellDofs(dof, user.mesh->getCellPtr(cellid));
  return q_array[dof[0]];
}

void GetDataNormal::get_vec(int id, Real * vec_out) const
{
  Point const* point = user.mesh->getNodePtr(id);
  vector<int> dofs(user.dim);

  user.getNodeDofs(&*point, DH_MESH, VAR_M, dofs.data());

  for (int i = 0; i < user.dim; ++i)
    vec_out[i] = q_array[*(dofs.data()+i)];
}

void GetDataMeshVel::get_vec(int id, Real * vec_out) const
{
  Point const* point = user.mesh->getNodePtr(id);
  vector<int> dofs(user.dim);

  user.getNodeDofs(&*point, DH_MESH, VAR_M, dofs.data());

  for (int i = 0; i < user.dim; ++i)
    vec_out[i] = q_array[*(dofs.data()+i)];
}

int GetDataCellTag::get_data_i(int cellid) const
{
  // assume que só há 1 grau de liberdade na célula
  return user.mesh->getCellPtr(cellid)->getTag();
}

double GetDataSlipVel::get_data_r(int nodeid) const
{
  Point const*const point = user.mesh->getNodePtr(nodeid);
  int dof;
  user.getNodeDofs(&*point, DH_SLIP, VAR_S, &dof);
  //user.dof_handler[DH_SLIP].getVariable(VAR_S).getVertexDofs(&dof, point);
  return q_array[dof];
}

/* petsc functions*/
//extern PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
//extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
//extern PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void *);
//extern PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
//extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
//extern PetscErrorCode FormJacobian_fs(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
//extern PetscErrorCode FormFunction_fs(SNES,Vec,Vec,void*);

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **argv)
{
  // initialization of FEPiCpp //////////////////////////////////////////////////
  printf("FEPiCpp version... 2018\n\n");
  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

  bool help_return, erro;
  AppCtx user(argc, argv, help_return, erro);
  if (help_return)
    return 0;
  if (erro)
    return 1;

  // if parallel //////////////////////////////////////////////////
#ifdef FEP_HAS_OPENMP
  {
    int nthreads = 0;
    #pragma omp parallel
    {
      #pragma omp critical
      nthreads = omp_get_num_threads(); //the number of threads is either the maximum of CPU cores (hyper-threading)
    }                                   //or the value of the global variable OMP_NUM_THREADS
    printf("\nOpenMP version.\n");
    printf("Num. threads: %d\n", nthreads);
  }
#else
  {
    printf("Serial version.\n");
  }
#endif

  // mesh and dofs treatment //////////////////////////////////////////////////
  user.loadMesh();
  user.loadDofs();  //counts dofs for U,P,V in cell, facet, corner
  user.evaluateQuadraturePts();
  erro = user.err_checks(); if (erro) return 1;

  // print info //////////////////////////////////////////////////
  user.printProblemInfo();
  user.onUpdateMesh();  //allocates Petsc objects as well as the matrix (jacobian) of the system (coloring)

  // solve time problem //////////////////////////////////////////////////
  user.solveTimeProblem();

  // free memory and finalizing //////////////////////////////////////////////////
  cout << "\n";
  user.timer.printTimes();
  user.freePetscObjs();
  PetscFinalize();
  //cout << "\a" << endl;
  return 0;
}
/* ------------------------------------------------------------------- */

/* ------------------------------------------------------------------- */
#if (false)
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"

PetscErrorCode FormJacobian(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formJacobian(snes,Vec_up_1,Mat_Jac,prejac,flag);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunction"

PetscErrorCode FormFunction(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formFunction(snes,Vec_up_1,Vec_fun);
  PetscFunctionReturn(0);
}
#endif
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "CheckSnesConvergence"

PetscErrorCode CheckSnesConvergence(SNES snes, PetscInt it,PetscReal xnorm, PetscReal pnorm, PetscReal fnorm, SNESConvergedReason *reason, void *ctx)
{
  AppCtx *user    = static_cast<AppCtx*>(ctx);
  user->checkSnesConvergence(snes, it, xnorm, pnorm, fnorm, reason);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian_mesh"

PetscErrorCode FormJacobian_mesh(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formJacobian_mesh(snes,Vec_up_1,Mat_Jac,prejac,flag);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunction_mesh"

PetscErrorCode FormFunction_mesh(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formFunction_mesh(snes,Vec_up_1,Vec_fun);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian_fs"

PetscErrorCode FormJacobian_fs(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formJacobian_fs(snes,Vec_up_1,Mat_Jac,prejac,flag);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunction_fs"

PetscErrorCode FormFunction_fs(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formFunction_fs(snes,Vec_up_1,Vec_fun);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian_sqrm"

PetscErrorCode FormJacobian_sqrm(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formJacobian_sqrm(snes,Vec_up_1,Mat_Jac,prejac,flag);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunction_mesh"

PetscErrorCode FormFunction_sqrm(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formFunction_sqrm(snes,Vec_up_1,Vec_fun);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian_fd"

PetscErrorCode FormJacobian_fd(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formJacobian_fd(snes,Vec_up_1,Mat_Jac,prejac,flag);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunction_fd"

PetscErrorCode FormFunction_fd(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr)
{
  AppCtx *user    = static_cast<AppCtx*>(ptr);
  user->formFunction_fd(snes,Vec_up_1,Vec_fun);
  PetscFunctionReturn(0);
}
