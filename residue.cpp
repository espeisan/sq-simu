#include "common.hpp"

#define CONTRACT1(i,       size_i)                         for (int i = 0; i < size_i; ++i)
#define CONTRACT2(i,j,     size_i, size_j)                 for (int i = 0; i < size_i; ++i) for (int j = 0; j < size_j; ++j)
#define CONTRACT3(i,j,k,   size_i, size_j, size_k)         for (int i = 0; i < size_i; ++i) for (int j = 0; j < size_j; ++j) for (int k = 0; k < size_k; ++k)
#define CONTRACT4(i,j,k,l, size_i, size_j, size_k, size_l) for (int i = 0; i < size_i; ++i) for (int j = 0; j < size_j; ++j) for (int k = 0; k < size_k; ++k) for (int l = 0; l < size_l; ++l)



int epsilon(int i, int j, int k)  // permutation function
{
  if(i==1 && j==2 && k==3) return  1;
  if(i==2 && j==3 && k==1) return  1;
  if(i==3 && j==1 && k==2) return  1;
  if(i==3 && j==2 && k==1) return -1;
  if(i==1 && j==3 && k==2) return -1;
  if(i==2 && j==1 && k==3) return -1;
return 0;
}

template<class D, class T>
D determinant(T const& a, int dim)  //determinant
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

template<class TensorType, class Double>  //inverse matrix
void invert_a(TensorType & a, int dim)
{
  if (dim==1)
  {
    a(0,0)=1./a(0,0);
  }
  else
  if (dim==2)
  {
    Double const det = a(0,0)*a(1,1)-a(0,1)*a(1,0);

    Double const inv00 = a(1,1)/det;
    Double const inv01 = -a(0,1)/det;
    Double const inv10 = -a(1,0)/det;
    Double const inv11 = a(0,0)/det;

    a(0,0) = inv00;
    a(0,1) = inv01;
    a(1,0) = inv10;
    a(1,1) = inv11;
  }
  else if (dim==3)
  {
    Double const det = a(0,0)*(a(1,1)*a(2,2)-a(1,2)*a(2,1))+a(0,1)*(a(1,2)*a(2,0)-a(1,0)*a(2,2))+a(0,2)*(a(1,0)*a(2,1)-a(1,1)*a(2,0));

    Double const inv00 = ( a(1,1)*a(2,2)-a(1,2)*a(2,1) )/det;
    Double const inv01 = ( a(0,2)*a(2,1)-a(0,1)*a(2,2) )/det;
    Double const inv02 = ( a(0,1)*a(1,2)-a(0,2)*a(1,1) )/det;
    Double const inv10 = ( a(1,2)*a(2,0)-a(1,0)*a(2,2) )/det;
    Double const inv11 = ( a(0,0)*a(2,2)-a(0,2)*a(2,0) )/det;
    Double const inv12 = ( a(0,2)*a(1,0)-a(0,0)*a(1,2) )/det;
    Double const inv20 = ( a(1,0)*a(2,1)-a(1,1)*a(2,0) )/det;
    Double const inv21 = ( a(0,1)*a(2,0)-a(0,0)*a(2,1) )/det;
    Double const inv22 = ( a(0,0)*a(1,1)-a(0,1)*a(1,0) )/det;

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

// ====================================================================================================
// to apply boundary conditions locally on NS problem.
template <typename Derived>
void getProjectorMatrix(MatrixBase<Derived> & P, int n_nodes, int const* nodes, Vec const& Vec_x_, double t, AppCtx const& app)
{
  int const dim = app.dim;
  Mesh const* mesh = &*app.mesh;
  //DofHandler const* dof_handler = &*app.dof_handler;
  std::vector<int> const& dirichlet_tags  = app.dirichlet_tags;
  //std::vector<int> const& neumann_tags    = app.neumann_tags  ;
  //std::vector<int> const& interface_tags  = app.interface_tags;
  std::vector<int> const& solid_tags      = app.solid_tags;
  std::vector<int> const& triple_tags     = app.triple_tags;
  //std::vector<int> const& periodic_tags   = app.periodic_tags ;
  std::vector<int> const& feature_tags    = app.feature_tags;
  std::vector<int> const& flusoli_tags    = app.flusoli_tags;
  //std::vector<int> const& solidonly_tags  = app.solidonly_tags;
  std::vector<int> const& slipvel_tags    = app.slipvel_tags;
  Vec const& Vec_normal = app.Vec_normal;

  P.setIdentity();

  Tensor I(dim,dim);
  Tensor Z(dim,dim);
  Vector X(dim);
  Vector normal(dim);
  int    dofs[dim];
  int    tag;
  Point const* point;

  I.setIdentity();
  Z.setZero();

  // NODES
  for (int i = 0; i < n_nodes; ++i)
  {
    point = mesh->getNodePtr(nodes[i]);
    tag = point->getTag();  //cout << tag << endl;

    if (is_in(tag,feature_tags))
    {
      app.getNodeDofs(&*point, DH_MESH, VAR_M, dofs);
      VecGetValues(Vec_x_, dim, dofs, X.data());
      P.block(i*dim,i*dim,dim,dim) = feature_proj(X,t,tag);
    }
    else if (is_in(tag,solid_tags) || is_in(tag, triple_tags))
    {
      app.getNodeDofs(&*point, DH_MESH, VAR_M, dofs);
      VecGetValues(Vec_x_, dim, dofs, X.data());
      normal = solid_normal(X,t,tag);
      P.block(i*dim,i*dim,dim,dim) = I - normal*normal.transpose();
      //P.block(i*dim,i*dim,dim,dim) = Z;
    }
    else if (is_in(tag,dirichlet_tags) || is_in(tag,slipvel_tags)) //solid motion and pure Dirichlet BC
    {
      P.block(i*dim,i*dim,dim,dim) = Z;
    }
    else if (is_in(tag,flusoli_tags)) //non-penetration BC
    {
      app.getNodeDofs(&*point, DH_MESH, VAR_M, dofs);
      VecGetValues(Vec_normal, dim, dofs, normal.data());
      P.block(i*dim,i*dim,dim,dim) = I - normal*normal.transpose(); //cout << endl << P << endl << endl << dofs[0] << ", " << dofs[1] << endl;
      if (app.is_axis && mesh->inBoundary(&*point)){
        VecGetValues(Vec_x_, dim, dofs, X.data());
        P.block(i*dim,i*dim,dim,dim) = feature_proj(X,t,tag)*P.block(i*dim,i*dim,dim,dim);
      }
    }
  } // end nodes

} // end getProjectorMatrix

// ====================================================================================================
// to apply boundary conditions on linear elasticity problem.
template <typename Derived>
void getProjectorBC(MatrixBase<Derived> & P, int n_nodes, int const* nodes, Vec const& Vec_x_, double t, AppCtx const& app)
{
  int const dim = app.dim;
  Mesh const* mesh = &*app.mesh;
  //DofHandler const* dof_handler = &*app.dof_handler;
  std::vector<int> const& dirichlet_tags  = app.dirichlet_tags;
  std::vector<int> const& neumann_tags    = app.neumann_tags  ;
  std::vector<int> const& interface_tags  = app.interface_tags;
  std::vector<int> const& solid_tags      = app.solid_tags    ;
  std::vector<int> const& triple_tags     = app.triple_tags   ;
  std::vector<int> const& periodic_tags   = app.periodic_tags ;
  std::vector<int> const& feature_tags    = app.feature_tags  ;
  std::vector<int> const& flusoli_tags    = app.flusoli_tags;
  std::vector<int> const& solidonly_tags  = app.solidonly_tags;
  std::vector<int> const& slipvel_tags    = app.slipvel_tags;
  Vec const& Vec_normal = app.Vec_normal;

  P.setIdentity();

  Tensor I(dim,dim);
  Tensor Z(dim,dim);
  Vector X(dim);
  Vector normal(dim);
  int    dofs[dim];
  int    tag;
  Point const* point;

  I.setIdentity();
  Z.setZero();

  bool boundary_smoothing = app.boundary_smoothing;

  // NODES
  for (int i = 0; i < n_nodes; ++i)
  {
    point = mesh->getNodePtr(nodes[i]);
    tag = point->getTag();

    if (is_in(tag,feature_tags))
    {
      if (boundary_smoothing)
      {      
        app.getNodeDofs(&*point, DH_MESH, VAR_M, dofs);
        VecGetValues(Vec_x_, dim, dofs, X.data());
        P.block(i*dim,i*dim,dim,dim)  = feature_proj(X,t,tag);
      }
      else
      {
        P.block(i*dim,i*dim,dim,dim) = Z;
      }
    }
    else if (is_in(tag,solid_tags) )
    {
      if (boundary_smoothing)
      {
        app.getNodeDofs(&*point, DH_MESH, VAR_M, dofs);
        VecGetValues(Vec_x_, dim, dofs, X.data());
        normal = -solid_normal(X,t,tag);
        P.block(i*dim,i*dim,dim,dim)  = I - normal*normal.transpose();
      }
      else
      {
        P.block(i*dim,i*dim,dim,dim) = Z;
      }
    }
    else if (is_in(tag,interface_tags))
    {
      if (false && boundary_smoothing)
      {
        app.getNodeDofs(&*point, DH_MESH, VAR_M, dofs);
        VecGetValues(Vec_normal, dim, dofs, X.data());
        P.block(i*dim,i*dim,dim,dim) = I - X*X.transpose();
      }
      else
      {
        P.block(i*dim,i*dim,dim,dim) = Z;
      }
    }
    else if (is_in(tag,triple_tags) || is_in(tag,dirichlet_tags) || is_in(tag,neumann_tags) || is_in(tag,periodic_tags)
                                    || is_in(tag,feature_tags)   || is_in(tag,flusoli_tags) || is_in(tag,solidonly_tags)
                                    || is_in(tag,slipvel_tags))
    {
      P.block(i*dim,i*dim,dim,dim) = Z;
    }
  } // end nodes
}// end getProjectorBC

// ====================================================================================================
// to apply boundary conditions on squirmer problem.
template <typename Derived>
void getProjectorSQRM(MatrixBase<Derived> & P, int n_nodes, int const* nodes, AppCtx const& app)
{
  Mesh const* mesh = &*app.mesh;
  //DofHandler const* dof_handler = &*app.dof_handler;
  std::vector<int> const& dirichlet_tags  = app.dirichlet_tags;
  std::vector<int> const& solidonly_tags  = app.solidonly_tags;

  P.setIdentity();

  int    tag;
  Point const* point;


  // NODES
  for (int i = 0; i < n_nodes; ++i)
  {
    point = mesh->getNodePtr(nodes[i]);
    tag = point->getTag();
    //m = point->getPosition() - mesh->numVerticesPerCell();
    //cell = mesh->getCellPtr(point->getIncidCell());

    if ( is_in(tag,dirichlet_tags) || is_in(tag,solidonly_tags) )
    {
      P(i,i) = 0.0;
    }

  } // end nodes
}

// ====================================================================================================
// to apply boundary conditions locally on dissipative force problem.
template <typename Derived>
void getProjectorFD(MatrixBase<Derived> & P, int n_nodes, int const* nodes, Vec const& Vec_x_, double t, AppCtx const& app)
{
  int const dim = app.dim;
  Mesh const* mesh = &*app.mesh;
  //DofHandler const* dof_handler = &*app.dof_handler;
  //std::vector<int> const& dirichlet_tags  = app.dirichlet_tags;
  //std::vector<int> const& neumann_tags    = app.neumann_tags  ;
  //std::vector<int> const& interface_tags  = app.interface_tags;
  //std::vector<int> const& solid_tags      = app.solid_tags;
  //std::vector<int> const& triple_tags     = app.triple_tags;
  //std::vector<int> const& periodic_tags   = app.periodic_tags ;
  //std::vector<int> const& feature_tags    = app.feature_tags;
  std::vector<int> const& flusoli_tags    = app.flusoli_tags;
  //std::vector<int> const& solidonly_tags  = app.solidonly_tags;
  std::vector<int> const& slipvel_tags    = app.slipvel_tags;
  //Vec const& Vec_normal = app.Vec_normal;

  P.setIdentity();

  Tensor I(dim,dim);
  Tensor Z(dim,dim);
  Vector X(dim);
  Vector normal(dim);
  //int    dofs[dim];
  int    tag;
  Point const* point;

  I.setIdentity();
  Z.setZero();

  // NODES
  for (int i = 0; i < n_nodes; ++i)
  {
    point = mesh->getNodePtr(nodes[i]);
    tag = point->getTag();  //cout << tag << endl;

    if (is_in(tag,flusoli_tags) || is_in(tag,slipvel_tags)) //non-penetration BC
    {
      P.block(i*dim,i*dim,dim,dim) = I; //cout << endl << P << endl << endl << dofs[0] << ", " << dofs[1] << endl;
    }
    else
    {
      P.block(i*dim,i*dim,dim,dim) = Z;
    }
  } // end nodes

} // end getProjectorMatrix

// ====================================================================================================
// to apply solid DOFS elimination to NS problem.
template <typename Derived>
void getProjectorDOFS(MatrixBase<Derived> & P, int n_nodes, int const* nodes, AppCtx const& app)
{
  int const LZ = app.LZ;
  //int const n_cell = n_nodes/app.dim;

  P.setZero();

  for (int i = 0; i < n_nodes; i++){
    for (int l = 0; l < LZ; l++){
      P(i*LZ + l,i*LZ + l) = nodes[l];
    }
  }

} // end getProjectorMatrix


// ******************************************************************************
//                            FORM FUNCTION_MESH
// ******************************************************************************
PetscErrorCode AppCtx::formFunction_mesh(SNES /*snes_m*/, Vec Vec_v, Vec Vec_fun)
{
  double utheta = AppCtx::utheta;  //cout << "mesh" << endl;

  // NOTE: solve elasticity problem in the mesh at time step n
  // NOTE: The mesh used is the Vec_x_0 or Vec_x_1, look for MESH CHOISE
  // WARNING: this function assumes that the boundary conditions was already applied

  Mat *JJ = &Mat_Jac_m;
  VecZeroEntries(Vec_fun);
  MatZeroEntries(*JJ);

// LOOP NAS CÉLULAS Parallel (uncomment it)
#ifdef FEP_HAS_OPENMP
  FEP_PRAGMA_OMP(parallel default(none) shared(Vec_v, Vec_fun, cout, JJ, utheta))
#endif
  {
    bool const non_linear = nonlinear_elasticity;

    Tensor            dxV(dim,dim);  // grad u
    Tensor            F_c(dim,dim);
    Tensor            invF_c(dim,dim);
    Tensor            invFT_c(dim,dim);
    Vector            Vqp(dim);
    MatrixXd          v_coefs_c_trans(dim, nodes_per_cell);  // mesh velocity;
    MatrixXd          v_coefs_c(nodes_per_cell, dim);
    MatrixXd          x_coefs_c_trans(dim, nodes_per_cell);
    MatrixXd          x_coefs_c(nodes_per_cell, dim);
    //MatrixXd          x_coefs_c_new_trans(dim, nodes_per_cell);
    //MatrixXd          x_coefs_c_new(nodes_per_cell, dim);
    MatrixXd          dxqsi_c(nodes_per_cell, dim);
    double            J, weight, JxW, MuE, LambE, ChiE, Jx0;

    VectorXd          Floc(n_dofs_v_per_cell);
    MatrixXd          Aloc(n_dofs_v_per_cell, n_dofs_v_per_cell);

    VectorXi          mapV_c(n_dofs_v_per_cell); //mapU_c(n_dofs_u_per_cell); // i think is n_dofs_v_per_cell

    MatrixXd          Prj(n_dofs_v_per_cell, n_dofs_v_per_cell);
    VectorXi          cell_nodes(nodes_per_cell);

    double            sigma_ck;
    double            dsigma_ckjd;

    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    cell_iterator cell = mesh->cellBegin(tid,nthreads);   //cell_iterator cell = mesh->cellBegin();
    cell_iterator cell_end = mesh->cellEnd(tid,nthreads); //cell_iterator cell_end = mesh->cellEnd();
    for (; cell != cell_end; ++cell)
    {

      // mapeamento do local para o global: (ID local para ID global)
      // mapV_c saves global IDs for cell's nodes unknowns enumeration
      dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapV_c.data(), &*cell);  //cout << mapV_c.transpose() << endl;
      //dof_handler[DH_UNKM].getVariable(VAR_U).getCellDofs(mapU_c.data(), &*cell);  //cout << mapU_c.transpose() << endl;
      /* Pega os valores das variáveis nos graus de liberdade */
      VecGetValues(Vec_v ,  mapV_c.size(), mapV_c.data(), v_coefs_c.data());  //cout << v_coefs_c << endl;//VecView(Vec_v,PETSC_VIEWER_STDOUT_WORLD);
      VecGetValues(Vec_x_1, mapV_c.size(), mapV_c.data(), x_coefs_c.data());  //cout << x_coefs_c << endl;// MESH CHOISE
      //VecGetValues(Vec_x_1, mapV_c.size(), mapV_c.data(), x_coefs_c_new.data());  //cout << x_coefs_c_new << endl;

      v_coefs_c_trans = v_coefs_c.transpose();
      x_coefs_c_trans = x_coefs_c.transpose();

      Floc.setZero();
      Aloc.setZero();

      // Quadrature
      for (int qp = 0; qp < n_qpts_cell; ++qp)
      {
        F_c = x_coefs_c_trans * dLqsi_c[qp];  //cout << dLqsi_c[qp] << endl;
        inverseAndDet(F_c,dim,invF_c,J);
        invFT_c= invF_c.transpose();  //usado?

        dxqsi_c = dLqsi_c[qp] * invF_c;

        dxV  = v_coefs_c_trans * dxqsi_c;       // n+utheta
        Vqp  = v_coefs_c_trans * qsi_c[qp];
        //Xqp      = x_coefs_c_trans * qsi_c[qp]; // coordenada espacial (x,y,z) do ponto de quadratura

        weight = quadr_cell->weight(qp);
        JxW = J*weight;  //parece que no es necesario, ver 2141 (JxW/JxW)
        MuE = 1*1.0/(pow(JxW,1.0));  LambE = 1*1.0/(pow(JxW,1.0));  ChiE = 0.0;  Jx0 = 1.0;

        for (int i = 0; i < n_dofs_v_per_cell/dim; ++i)  //sobre cantidad de funciones de forma
        {
          for (int c = 0; c < dim; ++c)  //sobre dimension
          {
            for (int k = 0; k < dim; ++k)  //sobre dimension
            {
              sigma_ck = dxV(c,k) + dxV(k,c); //sigma_ck = dxV(c,k);

              if (non_linear)  //is right?
              {
                for (int l = 0; l < dim; ++l)
                {
                  sigma_ck += dxV(l,c)*dxV(l,k);
                  if (c==k)
                  {
                    sigma_ck -= dxV(l,l);
                    for (int m = 0; m < dim; ++m)
                      sigma_ck -=  dxV(l,m)*dxV(l,m);
                  }
                }
              }  //end non_linear

              Floc(i*dim + c) += sigma_ck*dxqsi_c(i,k)*(pow(Jx0/JxW,ChiE)*JxW*MuE) + dxV(k,k)*dxqsi_c(i,c)*(pow(Jx0/JxW,ChiE)*JxW*LambE); // (JxW/JxW) is to compiler not complain about unused variables
              //Floc(i*dim + c) += sigma_ck*dxqsi_c(i,k)*(JxW*MuE) + dxV(k,k)*dxqsi_c(i,c)*(JxW*(MuE+LambE));
              for (int j = 0; j < n_dofs_v_per_cell/dim; ++j)
              {
                for (int d = 0; d < dim; ++d)
                {
                  dsigma_ckjd = 0;

                  if (c==d)
                    dsigma_ckjd = dxqsi_c(j,k);

                  if (k==d)
                    dsigma_ckjd += dxqsi_c(j,c);

                  if (non_linear)  //is right?
                  {
                    for (int l = 0; l < dim; ++l)
                    {
                      if (l==d)
                        dsigma_ckjd += dxqsi_c(j,c)*dxV(l,k) + dxV(l,c)*dxqsi_c(j,k);  //is ok?

                      if (c==k)
                      {
                        if (l==d)
                        {
                          dsigma_ckjd -= dxqsi_c(j,l);
                          for (int m = 0; m < dim; ++m)
                            dsigma_ckjd -= 2.*dxqsi_c(j,m)*dxV(l,m);
                        }
                      }
                    }
                  }  //end non_linear

                  Aloc(i*dim + c, j*dim + d) += dsigma_ckjd*dxqsi_c(i,k)*(pow(Jx0/JxW,ChiE)*JxW*MuE) + (1/dim)*dxqsi_c(j,d)*dxqsi_c(i,c)*(pow(Jx0/JxW,ChiE)*JxW*LambE);
                  //Aloc(i*dim + c, j*dim + d) += dsigma_ckjd*dxqsi_c(i,k)*(JxW*MuE) + (1/dim)*dxqsi_c(j,d)*dxqsi_c(i,c)*(JxW*(LambE+MuE));
                } // end d
              } // end j
            } // end k
          }// end c
        } // end i

      } // fim quadratura


      // Projection - to force non-penetrarion bc
      mesh->getCellNodesId(&*cell, cell_nodes.data());
      getProjectorBC(Prj, nodes_per_cell, cell_nodes.data(), Vec_x_1, current_time, *this /*AppCtx*/); // MESH CHOISE
      Floc = Prj*Floc;  //cout << Floc.transpose() << endl;
      Aloc = Prj*Aloc*Prj;  //zeros at dirichlet nodes (lines and columns)

#ifdef FEP_HAS_OPENMP
      FEP_PRAGMA_OMP(critical)
#endif
      {
        VecSetValues(Vec_fun, mapV_c.size(), mapV_c.data(), Floc.data(), ADD_VALUES);
        MatSetValues(*JJ, mapV_c.size(), mapV_c.data(), mapV_c.size(), mapV_c.data(), Aloc.data(), ADD_VALUES);
      }
    } // end cell loop


  } // end parallel
  //Assembly(*JJ); View(*JJ, "ElastOpAntes", "JJ");

  // boundary conditions on global Jacobian
    // solid & triple tags .. force normal
  if (true && force_dirichlet)  //identify the contribution of points in *_tags
  {
    int      nodeid;
    int      v_dofs[dim];
    Vector   normal(dim);
    Tensor   A(dim,dim);
    Tensor   I(Tensor::Identity(dim,dim));
    int      tag;

    point_iterator point = mesh->pointBegin();
    point_iterator point_end = mesh->pointEnd();
    for ( ; point != point_end; ++point)
    {
      tag = point->getTag();
      if (!(is_in(tag,feature_tags)   ||
            is_in(tag,solid_tags)     ||
            is_in(tag,interface_tags) ||
            is_in(tag,triple_tags)    ||
            is_in(tag,dirichlet_tags) ||
            is_in(tag,neumann_tags)   ||
            is_in(tag,periodic_tags)  ||
            is_in(tag,flusoli_tags)   ||
            is_in(tag,solidonly_tags) ||
            is_in(tag,slipvel_tags)   ))
        continue;
      //dof_handler[DH_UNKS].getVariable(VAR_U).getVertexAssociatedDofs(v_dofs, &*point);
      getNodeDofs(&*point, DH_MESH, VAR_M, v_dofs);

      nodeid = mesh->getPointId(&*point);
      getProjectorBC(A, 1, &nodeid, Vec_x_0, current_time, *this);
      A = I - A;
      MatSetValues(*JJ, dim, v_dofs, dim, v_dofs, A.data(), INSERT_VALUES);//ADD_VALUES);
    }
  }

  Assembly(*JJ); //View(*JJ, "matrizes/jac.m", "Jacm"); //MatView(*JJ,PETSC_VIEWER_STDOUT_WORLD);
  Assembly(Vec_fun);  //View(Vec_fun, "matrizes/rhs.m", "resm");
  //View(*JJ, "ElastOp", "JJ");
  //double val; VecNorm(Vec_fun,NORM_2,&val); cout << "norma residuo " << val <<endl;
  //cout << "Mesh calculation:" << endl;
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::formJacobian_mesh(SNES /*snes*/,Vec /*Vec_up_k*/,Mat* /**Mat_Jac*/, Mat* /*prejac*/, MatStructure * /*flag*/)
{
  // jacobian matrix is done in the formFunction_mesh
  PetscFunctionReturn(0);
}


// ******************************************************************************
//                            FORM FUNCTION_FS
// ******************************************************************************
PetscErrorCode AppCtx::formFunction_fs(SNES /*snes*/, Vec Vec_uzp_k, Vec Vec_fun_fs)
{
  double utheta = AppCtx::utheta;
  if (is_mr){utheta = 1.0;}

  VecSetOption(Vec_fun_fs, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  VecSetOption(Vec_uzp_k, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);

  std::vector<Vector3d> XG_mid = midGP(XG_1, XG_0, utheta, n_solids);

  int null_space_press_dof = -1;

  int iter;//, nodidd;
  SNESGetIterationNumber(snes_fs,&iter);  //cout << iter <<endl; //if starting newton iteration = niter, then iter = niter-1.

  if (!iter)
  {
    converged_times = 0;  //just for convergence test, it can be ignored
  }

  if (force_pressure)// && (iter<2))
  {
    Vector X(dim);
    Vector X_new(dim);
    if (behaviors & BH_Press_grad_elim)//TODO: what is BH_Press_grad_elim?
    {
      // fix the initial guess
      VecSetValue(Vec_uzp_k, null_space_press_dof, 0.0, INSERT_VALUES);
    }
    else
    {//imposes a pressure node (the first one in the mesh) at the Dirichlet region (or wherever you want)
      point_iterator point = mesh->pointBegin();
      while (!( mesh->isVertex(&*point) && (point->getTag() == 6)/*is_in(point->getTag(),dirichlet_tags) */) ){/*point->getTag() == 3*/
        ++point;
      }
      int x_dofs[3];
      dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(x_dofs, &*point);  //cout << x_dofs[0] << " " << x_dofs[1] << " " << x_dofs[2] << endl;
      VecGetValues(Vec_x_1, dim, x_dofs, X_new.data());
      VecGetValues(Vec_x_0, dim, x_dofs, X.data());
      X = .5*(X+X_new);
      dof_handler[DH_UNKM].getVariable(VAR_P).getVertexDofs(&null_space_press_dof, &*point); //null_space_press_dof used at the end for pressure imposition
      //cout << null_space_press_dof << endl;
      // fix the initial guess
      VecSetValue(Vec_uzp_k, null_space_press_dof, p_exact(X,current_time+.5*dt,point->getTag()), INSERT_VALUES);  //insert pressure value at node point
    }

    Assembly(Vec_uzp_k);

  }

  // checking if pressure value imposition at node point was successful.
  if (null_space_press_dof < 0 && force_pressure==1 && (iter<2))
  {
    cout << "force_pressure: something is wrong ..." << endl;
    throw;
  }

  Mat *JJ = &Mat_Jac_fs;
  VecZeroEntries(Vec_fun_fs);
  MatZeroEntries(*JJ);


  // LOOP NAS CÉLULAS Parallel (uncomment it) //////////////////////////////////////////////////
#ifdef FEP_HAS_OPENMP
  FEP_PRAGMA_OMP(parallel default(none) shared(Vec_uzp_k,Vec_fun_fs,cout,null_space_press_dof,JJ,utheta,iter,XG_mid))
#endif
  {
    VectorXd            FUloc(n_dofs_u_per_cell);  // U subvector part of F
    VectorXd            FPloc(n_dofs_p_per_cell);
    VectorXd            FZloc(n_dofs_z_per_cell);

    /* local data */
    int                 tag, tag_c, nod_id, nod_is, nod_sv, nodsum;
    MatrixXd            u_coefs_c_mid_trans(dim, n_dofs_u_per_cell/dim);  // n+utheta  // trans = transpost
    MatrixXd            u_coefs_c_old(n_dofs_u_per_cell/dim, dim);        // n
    MatrixXd            u_coefs_c_old_trans(dim,n_dofs_u_per_cell/dim);   // n
    MatrixXd            u_coefs_c_new(n_dofs_u_per_cell/dim, dim);        // n+1
    MatrixXd            u_coefs_c_new_trans(dim,n_dofs_u_per_cell/dim);   // n+1
    MatrixXd            uz_coefs_c(dim, n_dofs_u_per_cell/dim);
    MatrixXd            uz_coefs_c_old(dim, n_dofs_u_per_cell/dim);

    MatrixXd            du_coefs_c_old(n_dofs_u_per_cell/dim, dim);        // n
    MatrixXd            du_coefs_c_old_trans(dim,n_dofs_u_per_cell/dim);   // n
    MatrixXd            du_coefs_c_vold(n_dofs_u_per_cell/dim, dim);        // n-1
    MatrixXd            du_coefs_c_vold_trans(dim,n_dofs_u_per_cell/dim);   // n-1

    MatrixXd            u_coefs_c_om1(n_dofs_u_per_cell/dim, dim);        // n
    MatrixXd            u_coefs_c_om1_trans(dim,n_dofs_u_per_cell/dim);   // n
    MatrixXd            u_coefs_c_om2(n_dofs_u_per_cell/dim, dim);        // n-1
    MatrixXd            u_coefs_c_om2_trans(dim,n_dofs_u_per_cell/dim);   // n-1

    MatrixXd            u_coefs_c_om1c(n_dofs_u_per_cell/dim, dim);        // n
    MatrixXd            u_coefs_c_om1c_trans(dim,n_dofs_u_per_cell/dim);   // n
    MatrixXd            u_coefs_c_om2c(n_dofs_u_per_cell/dim, dim);        // n-1
    MatrixXd            u_coefs_c_om2c_trans(dim,n_dofs_u_per_cell/dim);   // n-1

    MatrixXd            v_coefs_c_mid(nodes_per_cell, dim);        // mesh velocity; n
    MatrixXd            v_coefs_c_mid_trans(dim,nodes_per_cell);   // mesh velocity; n

    VectorXd            p_coefs_c_new(n_dofs_p_per_cell);  // n+1
    VectorXd            p_coefs_c_old(n_dofs_p_per_cell);  // n
    VectorXd            p_coefs_c_mid(n_dofs_p_per_cell);  // n

    MatrixXd            z_coefs_c_mid_trans(LZ, n_dofs_z_per_cell/LZ);  // n+utheta  // trans = transpost
    MatrixXd            z_coefs_c_old(n_dofs_z_per_cell/LZ, LZ);        // n
    MatrixXd            z_coefs_c_old_trans(LZ,n_dofs_z_per_cell/LZ);   // n
    MatrixXd            z_coefs_c_new(n_dofs_z_per_cell/LZ, LZ);        // n+1
    MatrixXd            z_coefs_c_new_trans(LZ,n_dofs_z_per_cell/LZ);   // n+1
    MatrixXd            z_coefs_c_auo(dim, nodes_per_cell), z_coefs_c_aun(dim, nodes_per_cell);

    VectorXd            l_coefs_new(n_links);  // n+1

    MatrixXd            vs_coefs_c_mid_trans(dim, n_dofs_u_per_cell/dim);  // n+utheta  // trans = transpost
    MatrixXd            vs_coefs_c_old(n_dofs_u_per_cell/dim, dim);        // n
    MatrixXd            vs_coefs_c_old_trans(dim,n_dofs_u_per_cell/dim);   // n
    MatrixXd            vs_coefs_c_new(n_dofs_u_per_cell/dim, dim);        // n+1
    MatrixXd            vs_coefs_c_new_trans(dim,n_dofs_u_per_cell/dim);   // n+1
    MatrixXd            vs_coefs_c_om1(n_dofs_u_per_cell/dim, dim);        // n
    MatrixXd            vs_coefs_c_om1_trans(dim,n_dofs_u_per_cell/dim);   // n
    MatrixXd            vs_coefs_c_om2(n_dofs_u_per_cell/dim, dim);        // n-1
    MatrixXd            vs_coefs_c_om2_trans(dim,n_dofs_u_per_cell/dim);   // n-1

    MatrixXd            x_coefs_c_mid_trans(dim, nodes_per_cell); // n+utheta
    MatrixXd            x_coefs_c_new(nodes_per_cell, dim);       // n+1
    MatrixXd            x_coefs_c_new_trans(dim, nodes_per_cell); // n+1
    MatrixXd            x_coefs_c_old(nodes_per_cell, dim);       // n
    MatrixXd            x_coefs_c_old_trans(dim, nodes_per_cell); // n

    Tensor              F_c_mid(dim,dim);       // n+utheta
    Tensor              invF_c_mid(dim,dim);    // n+utheta
    //Tensor              invFT_c_mid(dim,dim);   // n+utheta

    Tensor              F_c_old(dim,dim);       // n
    Tensor              invF_c_old(dim,dim);    // n
    //Tensor              invFT_c_old(dim,dim);   // n

    Tensor              F_c_new(dim,dim);       // n+1
    Tensor              invF_c_new(dim,dim);    // n+1
    //Tensor              invFT_c_new(dim,dim);   // n+1

    /* All variables are in (n+utheta) by default */
    MatrixXd            dxphi_c(n_dofs_u_per_cell/dim, dim);
    MatrixXd            dxphi_c_new(dxphi_c);
    MatrixXd            dxpsi_c(n_dofs_p_per_cell, dim);
    MatrixXd            dxpsi_c_new(dxpsi_c);
    MatrixXd            dxqsi_c(nodes_per_cell, dim);
    Vector              dxbble(dim);
    Vector              dxbble_new(dim);
    Tensor              dxU(dim,dim), dxZ(dim,dim);   // grad u
    Tensor              dxU_old(dim,dim);   // grad u
    Tensor              dxU_new(dim,dim);   // grad u
    Tensor              dxUb(dim,dim);  // grad u bble
    Vector              dxP_new(dim);   // grad p
    Vector              Xqp(dim);
    Vector              Xqp_old(dim);
    Vector              Xc(dim);  // cell center; to compute CR element
    Vector              Uqp(dim), Zqp(dim);
    Vector              Ubqp(dim); // bble
    Vector              Uqp_old(dim), Zqp_old(dim); // n
    Vector              Uqp_new(dim), Zqp_new(dim); // n+1
    Vector              dUqp_old(dim), Uqp_m1(dim);  // n
    Vector              dUqp_vold(dim), Uqp_m2(dim);  // n
    Vector              Vqp(dim);
    Vector              Uconv_qp(dim);
    Vector              dUdt(dim);
    double              Pqp_new;
    VectorXi            cell_nodes(nodes_per_cell);
    double              J_mid;
    double              J_new, J_old;
    double              JxW_mid;  //JxW_new, JxW_old;
    double              weight;
    double              visc=-1; // viscosity
    double              cell_volume;
    double              hk2;
    double              tauk=0;
    double              delk=0;
    double              uconv;
    double              delta_cd;
    double              rho;
    double              ddt_factor;
    if (is_bdf2 && time_step > 0)
      ddt_factor = 1.5;
    else
    if (is_bdf3 && time_step > 1)
      ddt_factor = 11./6.;
    else
      ddt_factor = 1.;


    MatrixXd            Aloc(n_dofs_u_per_cell, n_dofs_u_per_cell);
    MatrixXd            Gloc(n_dofs_u_per_cell, n_dofs_p_per_cell);
    MatrixXd            Dloc(n_dofs_p_per_cell, n_dofs_u_per_cell);
    MatrixXd            Eloc(n_dofs_p_per_cell, n_dofs_p_per_cell);   // GSL, BC
    //MatrixXd            Cloc(n_dofs_u_per_cell, n_dofs_p_per_cell);   // GSL
    Tensor              iBbb(dim, dim);                               // BC, i : inverse ..it is not the inverse to CR element
    MatrixXd            Bbn(dim, n_dofs_u_per_cell);                  // BC
    MatrixXd            Bnb(n_dofs_u_per_cell, dim);                  // BC
    MatrixXd            Dpb(n_dofs_p_per_cell, dim);                  // BC
    MatrixXd            Gbp(dim, n_dofs_p_per_cell);                  // BC
    MatrixXd            Gnx(n_dofs_u_per_cell, dim);                  // CR ;; suffix x means p gradient
    Vector              FUb(dim);                                     // BC
    Vector              FPx(dim); // pressure gradient

    MatrixXd            Z1loc(n_dofs_u_per_cell,n_dofs_z_per_cell);
    MatrixXd            Z2loc(n_dofs_z_per_cell,n_dofs_u_per_cell);
    MatrixXd            Z3loc(n_dofs_z_per_cell,n_dofs_z_per_cell);
    MatrixXd            Z4loc(n_dofs_z_per_cell,n_dofs_p_per_cell);
    MatrixXd            Z5loc(n_dofs_p_per_cell,n_dofs_z_per_cell);

    MatrixXd            Asloc(n_dofs_u_per_cell, n_dofs_u_per_cell);
    MatrixXd            Dsloc(n_dofs_p_per_cell, n_dofs_u_per_cell);

    Vector              force_at_mid(dim);
    Vector              Res(dim), dRes(dim);                                     // residue
    Tensor              dResdu(dim,dim);                              // residue derivative
    Tensor const        Id(Tensor::Identity(dim,dim));
    Vector              vec(dim);     // temp
    Tensor              Ten(dim,dim); // temp

    VectorXi            mapU_c(n_dofs_u_per_cell);
    VectorXi            mapU_r(n_dofs_u_per_corner);
    VectorXi            mapP_c(n_dofs_p_per_cell);
    VectorXi            mapP_r(n_dofs_p_per_corner);
    VectorXi            mapZ_c(n_dofs_z_per_cell);
    VectorXi            mapZ_f(n_dofs_z_per_facet);
    VectorXi            mapZ_s(LZ);
    VectorXi            mapL_l(n_links);
    // mesh velocity
    VectorXi            mapM_c(dim*nodes_per_cell);
    //VectorXi            mapM_f(dim*nodes_per_facet);
    VectorXi            mapM_r(dim*nodes_per_corner);

    MatrixXd            Prj(n_dofs_u_per_cell,n_dofs_u_per_cell); // projector matrix
    //VectorXi            cell_nodes(nodes_per_cell);

    TensorZ const       IdZ(Tensor::Identity(LZ,LZ));  //TODO es Tensor::Indentity o TensorZ::Identity?
    MatrixXd            Id_vs(MatrixXd::Identity(n_dofs_u_per_cell,n_dofs_u_per_cell));
    VectorXi            mapU_t(n_dofs_u_per_cell), mapP_t(n_dofs_p_per_cell);
    VectorXi            mapUs_c(n_dofs_u_per_cell), mapUs_t(n_dofs_u_per_cell);

    //std::vector<bool>   SV(n_solids,false);       //slip velocity node history (Dirichlet BC)
    std::vector<int>    SV_c(nodes_per_cell,0);   //maximum nodes in slip visited saving the solid tag
    bool                SVI = false;              //slip velocity interaction, body pure Dirichlet node detected

    //std::vector<bool>   VS(n_solids,false);       //visited solid node history (Neumann BC)
    std::vector<int>    VS_c(nodes_per_cell,0);   //maximum nodes in visited solid saving the solid tag
    bool                VSF = false;              //visited solid-fluid, body mixed Neumann node detected
    bool                PDN = false;              //Pure Dirichlet Node at the fluid region found
    bool                FTN = false;              //Feature Tag Node found = true

    Vector   RotfI(dim), RotfJ(dim), ConfI(dim), ConfJ(dim);
    Vector3d XIg, XJg;
    Vector   XIp(dim), XIp_new(dim), XIp_old(dim), XJp(dim), XJp_new(dim), XJp_old(dim);
    Tensor   TenfI(dim,dim), TenfJ(dim,dim);
    Vector   auxRotf(dim), auxRotv(dim);
    Vector   auxRotvI(dim), auxRotvJ(dim);
    Tensor   auxTenf(dim,dim), auxTenv(dim,dim);
    Tensor   auxTenfI(dim,dim), auxTenfJ(dim,dim), auxTenvI(dim,dim), auxTenvJ(dim,dim);
    double   thetaI = 0.0;

    VectorXi            cell_nodes_tmp(nodes_per_cell);
    Tensor              F_c_curv(dim,dim);
    int                 tag_pt0, tag_pt1, tag_pt2, bcell, nPer, ccell;
    double const*       Xqpb;  //coordinates at the master element \hat{X}
    Vector              Phi(dim), DPhi(dim), X0(dim), X2(dim), T0(dim), T2(dim), Xcc(3), Vdat(3);
    bool                curvf = false;
    //Permutation matrices
    TensorXi            PerM3(TensorXi::Zero(3,3)), PerM6(TensorXi::Zero(6,6));
    MatrixXi            PerM12(MatrixXi::Zero(12,12));
    PerM3(0,1) = 1; PerM3(1,2) = 1; PerM3(2,0) = 1;
    PerM6(0,2) = 1; PerM6(1,3) = 1; PerM6(2,4) = 1; PerM6(3,5) = 1; PerM6(4,0) = 1; PerM6(5,1) = 1;
    PerM12.block(0,0,6,6) = PerM6; PerM12.block(6,6,6,6) = PerM6;
    double              areas = 0.0;
    Tensor              Lcil(Tensor::Zero(dim,dim));  //diagonal matrix with cylindrical laplacian

    ////////////////////////////////////////////////// STARTING CELL ITERATION //////////////////////////////////////////////////
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    cell_iterator cell = mesh->cellBegin(tid,nthreads);    //cell_iterator cell = mesh->cellBegin();
    cell_iterator cell_end = mesh->cellEnd(tid,nthreads);  //cell_iterator cell_end = mesh->cellEnd();

    for (; cell != cell_end; ++cell)
    {

      tag = cell->getTag();

      //considers the cells completely inside the solid to impose the Dirichlet Fluid-Solid condition ///////////////////////////
      if(is_in(tag,solidonly_tags)){
        FUloc.setZero();
        FPloc.setZero();
        Aloc.setIdentity();
        Eloc.setIdentity();
        Z1loc.setZero();

        u_coefs_c_new = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
        z_coefs_c_new = MatrixXd::Zero(n_dofs_z_per_cell/LZ,LZ);
        //vs_coefs_c_new = MatrixXd::Zero(n_dofs_u_per_cell/dim, dim);
        mapU_t = -VectorXi::Ones(n_dofs_u_per_cell);
        mapP_t = -VectorXi::Ones(n_dofs_p_per_cell);
        mapZ_c = -VectorXi::Ones(n_dofs_z_per_cell);
        //if (is_unksv){mapUs_c = -VectorXi::Ones(n_dofs_u_per_cell);}

        dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);
        dof_handler[DH_UNKM].getVariable(VAR_U).getCellDofs(mapU_c.data(), &*cell);
        dof_handler[DH_UNKM].getVariable(VAR_P).getCellDofs(mapP_c.data(), &*cell);
        VecGetValues(Vec_x_1,     mapM_c.size(), mapM_c.data(), x_coefs_c_new.data());
        VecGetValues(Vec_uzp_k,   mapU_c.size(), mapU_c.data(), u_coefs_c_new.data());
        VecGetValues(Vec_uzp_k,   mapP_c.size(), mapP_c.data(), p_coefs_c_new.data());
        //if (is_unksv){mapUs_c = -(mapU_c + n_unknowns_ups*VectorXi::Ones(n_dofs_u_per_cell));}

        x_coefs_c_new_trans = x_coefs_c_new.transpose();

        //correcting dofs for solid velocity nodes
        for (int i = 0; i < n_dofs_u_per_cell/dim; ++i)
        {
          tag_c  = mesh->getNodePtr(cell->getNodeId(i))->getTag();
          nod_is = is_in_id(tag_c,solidonly_tags);
          //nod_sv = is_in_id(tag_c,slipvel_tags);
          nodsum = nod_is; //+nod_sv;
          if (nodsum){
            for (int C = 0; C < LZ; C++){
              mapZ_c(i*LZ + C) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + C;
            }
            for (int l = 0; l < dim; l++){
              mapU_t(i*dim + l) = mapU_c(i*dim + l);
              //if (is_unksv){mapUs_c(i*dim + l) = -mapUs_c(i*dim + l)/*mapU_c(i*dim + l) + n_unknowns_ups*/;}
            }
          }
        }
        //correcting dofs for solid pressure nodes
        for (int i = 0; i < n_dofs_p_per_cell; ++i)
        {
          tag_c  = mesh->getNodePtr(cell->getNodeId(i))->getTag();
          nod_is = is_in_id(tag_c,solidonly_tags);
          //nod_sv = is_in_id(tag_c,slipvel_tags);
          if (dup_press_nod){
            nodsum = 1;//this enforces zero value for interf. internal pressure nodes in duplicated nodes case
          }
          else{
            nodsum = nod_is;//this enforces zero value ONLY for strict internal solid nodes
          }
          if (nodsum){
            mapP_t(i) = mapP_c(i);
            FPloc(i) = p_coefs_c_new(i) - 0.0; //set 0.0 pressure at interior solid only nodes
          }
        }

        VecGetValues(Vec_uzp_k,   mapZ_c.size(), mapZ_c.data(), z_coefs_c_new.data());
        z_coefs_c_new_trans = z_coefs_c_new.transpose();  //cout << z_coefs_c_new.transpose() << endl << endl;

        for (int i = 0; i < n_dofs_u_per_cell/dim; ++i){
          tag_c  = mesh->getNodePtr(cell->getNodeId(i))->getTag();
          nod_is = is_in_id(tag_c,solidonly_tags);
          nodsum = nod_is;
          if (nodsum == 0)
            continue;

          XIp    = x_coefs_c_new_trans.col(i); //ref point Xp, old, mid, or new
          XIg    = XG_mid[nodsum-1];                //mass center, mid, _0, "new"
          RotfI  = SolidVel(XIp, XIg, z_coefs_c_new_trans.col(i), dim);  //cout << RotfI << endl;
          //if (nod_sv){
          //  VecGetValues(Vec_slipv_1,  mapU_t.size(), mapU_t.data(), vs_coefs_c_new.data()); //cout << vs_coefs_c_new << endl << endl;
          //}
          for (int c = 0; c < dim; ++c){
            FUloc(i*dim + c) = u_coefs_c_new(i,c) - 1*RotfI(c);// - vs_coefs_c_new(i,c);
            for (int D = 0; D < LZ; D++){
              RotfJ  = SolidVel(XIp, XIg, IdZ.col(D), dim);
              Z1loc(i*dim+c,i*LZ+D) = -RotfJ(c);
            }
          }

        }//end for cell nodes

        #ifdef FEP_HAS_OPENMP
          FEP_PRAGMA_OMP(critical)
        #endif
        {//here we use ADD_VALUES and gives us the momentum equation line, times the quantity of elements touching each solid node (which is not a problem at all)
            VecSetValues(Vec_fun_fs, mapU_t.size(), mapU_t.data(), FUloc.data(), ADD_VALUES);  //cout << FUloc.transpose() << "   " << mapU_t.transpose() << endl;
            VecSetValues(Vec_fun_fs, mapP_t.size(), mapP_t.data(), FPloc.data(), ADD_VALUES);
            MatSetValues(*JJ, mapU_t.size(), mapU_t.data(), mapZ_c.size(), mapZ_c.data(), Z1loc.data(), ADD_VALUES);  //cout << Z1loc.transpose() << "   " << mapZ_c.transpose() << endl <<endl;
            MatSetValues(*JJ, mapU_t.size(), mapU_t.data(), mapU_t.size(), mapU_t.data(), Aloc.data(), ADD_VALUES);
            MatSetValues(*JJ, mapP_t.size(), mapP_t.data(), mapP_t.size(), mapP_t.data(), Eloc.data(), ADD_VALUES);
            //if (is_unksv){
            //  MatSetValues(*JJ, mapUs_c.size(), mapUs_c.data(), mapUs_c.size(), mapUs_c.data(), Aloc.data(), ADD_VALUES);
            //}
        }
        continue;
      }
      //nodes at the interface are considered for cells outside the solids //////////////////////////////////////////////////

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
      dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);  //cout << mapM_c.transpose() << endl;  //unk. global ID's
      dof_handler[DH_UNKM].getVariable(VAR_U).getCellDofs(mapU_c.data(), &*cell);  //cout << mapU_c.transpose() << endl;
      dof_handler[DH_UNKM].getVariable(VAR_P).getCellDofs(mapP_c.data(), &*cell);  //cout << mapP_c.transpose() << endl;

      if (is_sfip){// dofs for Z (or S) solid velocity variable, and correction of dof mapU_c
        mapZ_c = -VectorXi::Ones(n_dofs_z_per_cell);
        mapU_t = -VectorXi::Ones(n_dofs_u_per_cell);
        //mapUs_c = -VectorXi::Ones(n_dofs_u_per_cell);
        ///if (is_unksv){mapUs_c = -(mapU_c + n_unknowns_ups*VectorXi::Ones(n_dofs_u_per_cell))/*VectorXi::Ones(n_dofs_u_per_cell)*/;}
        SVI = false; VSF = false;
        for (int j = 0; j < n_dofs_u_per_cell/dim; ++j){
          tag_c = mesh->getNodePtr(cell_nodes(j)/*cell->getNodeId(j)*/)->getTag();
          nod_id = is_in_id(tag_c,flusoli_tags);
          //nod_is = 0;//is_in_id(tag_c,solidonly_tags);  //always zero: look the previous is_in(tag,solidonly_tags) condition
          nod_sv = is_in_id(tag_c,slipvel_tags);
          nodsum = nod_sv+nod_id; //TODO +nod_id before
          if (nodsum){
            for (int l = 0; l < LZ; l++){
              mapZ_c(j*LZ + l) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + l;
            }
            for (int l = 0; l < dim; l++){
              mapU_t(j*dim + l) = mapU_c(j*dim + l); //mapU_c(j*dim + l) = -1;
              //if (is_unksv){mapUs_c(j*dim + l) = mapU_c(j*dim + l) + n_unknowns_ups/*-mapUs_c(i*dim + l)*/;}
            }
          }
          if (nod_sv){SVI = true;}  //at least one sv node found
          if (nod_id){VSF = true;}  //at least one fsi node found //ojo antes solo nod_sv  //TODO nod_id+nod_sv
          SV_c[j] = nod_sv; VS_c[j] = nod_id;  //ojo antes solo nod_sv para VS_c
        }

        //if (is_unksv){mapUs_t = -mapUs_c;}
      //if (is_unksv && (SVI || VSF)){//slip vel map for boundary touching cells
      //  mapUs_c = mapU_c + n_unknowns_ups*VectorXi::Ones(n_dofs_u_per_cell);
      //}
      }
      //if cell is permuted, this corrects the maps; mapZ is already corrected by hand in the previous if conditional
      if (curvf){
        for (int l = 0; l < nPer; l++){
          mapM_c = PerM6*mapM_c;  //cout << mapM_c.transpose() << endl;
          mapU_c = PerM6*mapU_c;  //cout << mapU_c.transpose() << endl;
          mapP_c = PerM3*mapP_c;  //cout << mapP_c.transpose() << endl;
          mapU_t = PerM6*mapU_t;  //cout << mapU_t.transpose() << endl;
        }
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      //identify if the cell has at least one Pure Dirichlet node at the fluid region //////////////////////////////////////////////////
      PDN = false; FTN = false;
      for (int j = 0; j < n_dofs_u_per_cell/dim; ++j){
        tag_c = mesh->getNodePtr(cell_nodes(j)/*cell->getNodeId(j)*/)->getTag();
        if (is_in(tag_c,dirichlet_tags)){
          PDN = true;
        }
        else if (is_in(tag_c,feature_tags)){
          FTN = true;
        }
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      //initialize zero values for the variables//////////////////////////////////////////////////
      u_coefs_c_old = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      u_coefs_c_new = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      z_coefs_c_old = MatrixXd::Zero(n_dofs_z_per_cell/LZ,LZ);
      z_coefs_c_new = MatrixXd::Zero(n_dofs_z_per_cell/LZ,LZ);
      z_coefs_c_auo = MatrixXd::Zero(dim,nodes_per_cell);
      z_coefs_c_aun = MatrixXd::Zero(dim,nodes_per_cell);
      uz_coefs_c    = MatrixXd::Zero(dim,n_dofs_u_per_cell/dim);
      uz_coefs_c_old= MatrixXd::Zero(dim,n_dofs_u_per_cell/dim);

      du_coefs_c_old= MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      u_coefs_c_om1 = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      u_coefs_c_om1c= MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      if (is_bdf3){
        du_coefs_c_vold= MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
        u_coefs_c_om2  = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
        u_coefs_c_om2c = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      }

      vs_coefs_c_old = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      vs_coefs_c_new = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      vs_coefs_c_om1 = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      if (is_bdf3) vs_coefs_c_om2 = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      vs_coefs_c_mid_trans = MatrixXd::Zero(dim,nodes_per_cell);
      vs_coefs_c_old_trans = MatrixXd::Zero(dim,nodes_per_cell);
      vs_coefs_c_new_trans = MatrixXd::Zero(dim,nodes_per_cell);

      //get the value for the variables//////////////////////////////////////////////////
      if ((is_bdf2 && time_step > 0) || (is_bdf3 && time_step > 1))
        VecGetValues(Vec_v_1, mapM_c.size(), mapM_c.data(), v_coefs_c_mid.data());
      else
        VecGetValues(Vec_v_mid, mapM_c.size(), mapM_c.data(), v_coefs_c_mid.data());  //cout << v_coefs_c_mid << endl << endl;//size of vector mapM_c

      VecGetValues(Vec_x_0,     mapM_c.size(), mapM_c.data(), x_coefs_c_old.data());  //cout << x_coefs_c_old << endl << endl;
      VecGetValues(Vec_x_1,     mapM_c.size(), mapM_c.data(), x_coefs_c_new.data());  //cout << x_coefs_c_new << endl << endl;
      VecGetValues(Vec_uzp_0,   mapU_c.size(), mapU_c.data(), u_coefs_c_old.data());  //cout << u_coefs_c_old << endl << endl;
      VecGetValues(Vec_uzp_k,   mapU_c.size(), mapU_c.data(), u_coefs_c_new.data());  //cout << u_coefs_c_new << endl << endl;
      VecGetValues(Vec_uzp_0,   mapP_c.size(), mapP_c.data(), p_coefs_c_old.data());  //cout << p_coefs_c_old << endl << endl;
      VecGetValues(Vec_uzp_k,   mapP_c.size(), mapP_c.data(), p_coefs_c_new.data());  //cout << p_coefs_c_new << endl << endl;
      if (is_sfip){
        VecGetValues(Vec_uzp_0,   mapZ_c.size(), mapZ_c.data(), z_coefs_c_old.data());  //cout << z_coefs_c_old << endl << endl;
        VecGetValues(Vec_uzp_k,   mapZ_c.size(), mapZ_c.data(), z_coefs_c_new.data());  //cout << z_coefs_c_new << endl << endl;
      }

      VecGetValues(Vec_uzp_m1,  mapU_c.size(), mapU_c.data(), u_coefs_c_om1.data()); // bdf2,bdf3
      if (is_sfip) VecGetValues(Vec_uzp_m1,  mapU_t.size(), mapU_t.data(), u_coefs_c_om1c.data()); // bdf2,bdf3
      if (is_bdf3){
        VecGetValues(Vec_uzp_m2,  mapU_c.size(), mapU_c.data(), u_coefs_c_om2.data());
        if (is_sfip) VecGetValues(Vec_uzp_m2,  mapU_t.size(), mapU_t.data(), u_coefs_c_om2c.data()); // bdf3
      }

      if (!is_unksv && SVI){//TODO check if is necessary
        VecGetValues(Vec_slipv_0,  mapU_t.size(), mapU_t.data(), vs_coefs_c_old.data()); // bdf2,bdf3
        VecGetValues(Vec_slipv_1,  mapU_t.size(), mapU_t.data(), vs_coefs_c_new.data()); // bdf2,bdf3
        VecGetValues(Vec_slipv_m1, mapU_t.size(), mapU_t.data(), vs_coefs_c_om1.data()); // bdf2,bdf3
        if (is_bdf3)
          VecGetValues(Vec_slipv_m2,  mapU_t.size(), mapU_t.data(), vs_coefs_c_om2.data()); // bdf3
      }
      //else if (is_unksv && (SVI || VSF)){
      //  VecGetValues(Vec_uzp_0,  mapUs_c.size(), mapUs_c.data(), vs_coefs_c_old.data()); // bdf2,bdf3
      //  VecGetValues(Vec_uzp_k,  mapUs_c.size(), mapUs_c.data(), vs_coefs_c_new.data()); // bdf2,bdf3
      //}

      v_coefs_c_mid_trans = v_coefs_c_mid.transpose();  //cout << v_coefs_c_mid_trans << endl << endl;
      x_coefs_c_old_trans = x_coefs_c_old.transpose();
      x_coefs_c_new_trans = x_coefs_c_new.transpose();
      u_coefs_c_old_trans = u_coefs_c_old.transpose();  //cout << u_coefs_c_old_trans << endl << endl;
      u_coefs_c_new_trans = u_coefs_c_new.transpose();
      z_coefs_c_old_trans = z_coefs_c_old.transpose();  //cout << z_coefs_c_old_trans << endl << endl;
      z_coefs_c_new_trans = z_coefs_c_new.transpose();

      du_coefs_c_old_trans= du_coefs_c_old.transpose(); // bdf2
      u_coefs_c_om1_trans = u_coefs_c_om1.transpose();
      u_coefs_c_om1c_trans= u_coefs_c_om1c.transpose();
      if (is_bdf3){
        du_coefs_c_vold_trans= du_coefs_c_vold.transpose(); // bdf3
        u_coefs_c_om2_trans  = u_coefs_c_om2.transpose();
        u_coefs_c_om2c_trans = u_coefs_c_om2c.transpose();
      }

      x_coefs_c_mid_trans = utheta*x_coefs_c_new_trans + (1.-utheta)*x_coefs_c_old_trans;
      u_coefs_c_mid_trans = utheta*u_coefs_c_new_trans + (1.-utheta)*u_coefs_c_old_trans;
      p_coefs_c_mid       = utheta*p_coefs_c_new       + (1.-utheta)*p_coefs_c_old;
      z_coefs_c_mid_trans = utheta*z_coefs_c_new_trans + (1.-utheta)*z_coefs_c_old_trans;

      if (SVI){
        vs_coefs_c_old_trans = vs_coefs_c_old.transpose();  //cout << u_coefs_c_old_trans << endl << endl;
        vs_coefs_c_new_trans = vs_coefs_c_new.transpose();
        vs_coefs_c_om1_trans = vs_coefs_c_om1.transpose();
        if (is_bdf3)
          vs_coefs_c_om2_trans = vs_coefs_c_om2.transpose();

        vs_coefs_c_mid_trans = utheta*vs_coefs_c_new_trans + (1.-utheta)*vs_coefs_c_old_trans;
      }

      if (curvf){
        tag_pt0 = mesh->getNodePtr(cell_nodes(0))->getTag();
        nod_id = is_in_id(tag_pt0,flusoli_tags)+is_in_id(tag_pt0,slipvel_tags);
        Xcc = XG_0[nod_id-1];
        X0(0) = x_coefs_c_mid_trans(0,0);  X0(1) = x_coefs_c_mid_trans(1,0);
        X2(0) = x_coefs_c_mid_trans(0,2);  X2(1) = x_coefs_c_mid_trans(1,2);
        Vdat << RV[nod_id-1](0),RV[nod_id-1](1), 0.0; //theta_0[nod_id-1]; //container for R1, R2, theta
      }

      if (false && (function_space == P2P1)){
        for (int j = 3; j < 6; ++j){
          tag_c = mesh->getNodePtr(cell_nodes(j)/*cell->getNodeId(j)*/)->getTag();
          if (is_in(tag_c,flusoli_tags) || is_in(tag_c,slipvel_tags)){
            if (j == 3){
              X0(0) = x_coefs_c_mid_trans(0,0);  X0(1) = x_coefs_c_mid_trans(1,0);
              X2(0) = x_coefs_c_mid_trans(0,1);  X2(1) = x_coefs_c_mid_trans(1,1);
            }
            else if (j == 4){
              X0(0) = x_coefs_c_mid_trans(0,1);  X0(1) = x_coefs_c_mid_trans(1,1);
              X2(0) = x_coefs_c_mid_trans(0,2);  X2(1) = x_coefs_c_mid_trans(1,2);
            }
            else{
              X0(0) = x_coefs_c_mid_trans(0,2);  X0(1) = x_coefs_c_mid_trans(1,2);
              X2(0) = x_coefs_c_mid_trans(0,0);  X2(1) = x_coefs_c_mid_trans(1,0);
            }
            nod_id = is_in_id(tag_c,flusoli_tags)+is_in_id(tag_c,slipvel_tags);
            Xcc = XG_0[nod_id-1];
            Vdat << RV[nod_id-1](0),RV[nod_id-1](1), 0.0; //theta_0[nod_id-1]; //container for R1, R2, theta
            Phi = curved_Phi(0.5,X0,X2,Xcc,Vdat,dim);
            //cout << Phi.transpose() << endl; cout << x_coefs_c_mid_trans(0,j) << "," << x_coefs_c_mid_trans(1,j) << endl;
            //correcting the components of the node that goes to the interface
            x_coefs_c_mid_trans(0,j) += 0.5*Phi(0);
            x_coefs_c_mid_trans(1,j) += 0.5*Phi(1); //cout << x_coefs_c_mid_trans(0,j) << "," << x_coefs_c_mid_trans(1,j) << endl;
            break;
          }
        }
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      //viscosity and density at the cell//////////////////////////////////////////////////
      visc = muu(tag);
      rho  = pho(Xqp,tag);  //pho is elementwise, so Xqp does nothing
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      //initialization as zero of the residuals and elemental matrices////////////////////////////////////////
      FUloc.setZero();
      FPloc.setZero();
      FZloc.setZero();

      Aloc.setZero();
      Gloc.setZero();
      Dloc.setZero();
      Eloc.setZero();
      Z1loc.setZero(); Z2loc.setZero(); Z3loc.setZero(); Z4loc.setZero(); Z5loc.setZero();
      Asloc.setZero(); Dsloc.setZero();

      if (behaviors & BH_bble_condens_PnPn) //reset matrices
      {
        iBbb.setZero();
        Bnb.setZero();
        Gbp.setZero();
        FUb.setZero();
        Bbn.setZero();
        Dpb.setZero();
      }

      if(behaviors & BH_GLS) //GLS stabilization elemental matrices and residual initialization
      {
        cell_volume = 0;
        for (int qp = 0; qp < n_qpts_cell; ++qp) {
          F_c_mid = Tensor::Zero(dim,dim);  //Zero(dim,dim);
          if (curvf){//F_c_curv.setZero();
            Xqpb = quadr_cell->point(qp);
            Phi = curved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
            DPhi = Dcurved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
            F_c_curv.col(0) = -Phi;
            F_c_curv.col(1) = -Phi + (1.0-Xqpb[0]-Xqpb[1])*DPhi;
            F_c_mid = F_c_curv;
          }
          F_c_mid += x_coefs_c_mid_trans * dLqsi_c[qp];
          J_mid = determinant(F_c_mid,dim);
          if (is_axis && false){
            J_mid = J_mid*2*pi*Xqp(0);
          }
          cell_volume += J_mid * quadr_cell->weight(qp);
        }  //cout << J_mid << " " << cell_volume << endl;

        hk2 = cell_volume / pi; //element size: radius a circle with the same volume of the cell

        if (false && SVI){
          for (int J = 0; J < nodes_per_cell; J++){
            if (SV_c[J]){
              XJp_old = x_coefs_c_old_trans.col(J);
              uz_coefs_c_old.col(J) = SolidVel(XJp_old,XG_0[SV_c[J]-1],z_coefs_c_old_trans.col(J),dim);
              //if (VS_c[J])
                uz_coefs_c_old.col(J) += vs_coefs_c_old_trans.col(J);
            }
          }
        }
        uconv = (u_coefs_c_old - v_coefs_c_mid /*+ uz_coefs_c_old.transpose()*/).lpNorm<Infinity>();
        if (false && SVI){
          tauk = 4.*visc/hk2 + 2.*rho*uconv/sqrt(hk2);
          tauk = 1./tauk;
        }
        else{tauk = hk2/(12*visc);}
        if (dim==3)
          tauk *= 0.1;

        //delk = 4.*visc + 2.*rho*uconv*sqrt(hk2);
        delk = 0*4*visc;

        Eloc.setZero();
        //Cloc.setZero();
      }

      if (behaviors & BH_bble_condens_CR) //bubble stabilization
      {
//        bble_integ = 0;
        Gnx.setZero();
        iBbb.setZero();
        Bnb.setZero();
        FUb.setZero();
        FPx.setZero();
        Bbn.setZero();

        cell_volume = 0;
        Xc.setZero();
        for (int qp = 0; qp < n_qpts_cell; ++qp) {
          F_c_mid = x_coefs_c_mid_trans * dLqsi_c[qp];
          J_mid = determinant(F_c_mid,dim);
          Xqp  = x_coefs_c_mid_trans * qsi_c[qp];
          cell_volume += J_mid * quadr_cell->weight(qp);
          Xc += J_mid * quadr_cell->weight(qp) * Xqp;
        }
        Xc /= cell_volume;
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
      for (int qp = 0; qp < n_qpts_cell; ++qp)
      {

        F_c_mid = Tensor::Zero(dim,dim);  //Zero(dim,dim);
        F_c_old = Tensor::Zero(dim,dim);  //Zero(dim,dim);
        F_c_new = Tensor::Zero(dim,dim);  //Zero(dim,dim);
        Xqp     = Vector::Zero(dim);// coordenada espacial (x,y,z) do ponto de quadratura
        Xqp_old = Vector::Zero(dim); // coordenada espacial (x,y,z) do ponto de quadratura
        if (curvf){//F_c_curv.setZero();
          Xqpb = quadr_cell->point(qp);
          Phi = curved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
          DPhi = Dcurved_Phi(Xqpb[1],X0,X2,Xcc,Vdat,dim);
          F_c_curv.col(0) = -Phi;
          F_c_curv.col(1) = -Phi + (1.0-Xqpb[0]-Xqpb[1])*DPhi;
          F_c_mid = F_c_curv;
          F_c_old = F_c_curv;
          F_c_new = F_c_curv;
          Xqp     = (1.0-Xqpb[0]-Xqpb[1])*Phi;
          Xqp_old = (1.0-Xqpb[0]-Xqpb[1])*Phi;
        }

        F_c_mid += x_coefs_c_mid_trans * dLqsi_c[qp];  // (dim x nodes_per_cell) (nodes_per_cell x dim)
        F_c_old += x_coefs_c_old_trans * dLqsi_c[qp];
        F_c_new += x_coefs_c_new_trans * dLqsi_c[qp];
        inverseAndDet(F_c_mid,dim,invF_c_mid,J_mid);
        inverseAndDet(F_c_old,dim,invF_c_old,J_old);
        inverseAndDet(F_c_new,dim,invF_c_new,J_new);
        //invFT_c_mid = invF_c_mid.transpose();
        //invFT_c_old = invF_c_old.transpose();
        //invFT_c_new = invF_c_new.transpose();

        dxphi_c_new = dLphi_c[qp] * invF_c_new;
        dxphi_c     = dLphi_c[qp] * invF_c_mid;
        dxpsi_c_new = dLpsi_c[qp] * invF_c_new;
        dxpsi_c     = dLpsi_c[qp] * invF_c_mid;
        dxqsi_c     = dLqsi_c[qp] * invF_c_mid;

        dxU      = u_coefs_c_mid_trans * dLphi_c[qp] * invF_c_mid; // n+utheta
        dxU_new  = u_coefs_c_new_trans * dLphi_c[qp] * invF_c_new; // n+1
        dxU_old  = u_coefs_c_old_trans * dLphi_c[qp] * invF_c_old; // n
        dxP_new  = dxpsi_c.transpose() * p_coefs_c_new;

        Xqp     += x_coefs_c_mid_trans * qsi_c[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
        Xqp_old += x_coefs_c_old_trans * qsi_c[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
        Uqp      = u_coefs_c_mid_trans * phi_c[qp]; //n+utheta
        Uqp_new  = u_coefs_c_new_trans * phi_c[qp]; //n+1
        Uqp_old  = u_coefs_c_old_trans * phi_c[qp]; //n
        Pqp_new  = p_coefs_c_new.dot(psi_c[qp]);
//        Pqp      = p_coefs_c_mid.dot(psi_c[qp]);
        Vqp      = v_coefs_c_mid_trans * qsi_c[qp];
        dUdt     = (Uqp_new-Uqp_old)/dt;  //D1f^{n+1}/dt = U^{n+1}/dt-U^{n}/dt
        Uconv_qp = Uqp - Vqp;  //Uconv_qp = Uqp_old;

        if (is_bdf2 && time_step > 0)
        {
          Uqp_m1 = u_coefs_c_om1_trans * phi_c[qp];
          dUdt = 1.5*dUdt - 1./2.*Uqp_old/dt + 1./2.*Uqp_m1/dt;
          //dUqp_old  = du_coefs_c_old_trans * phi_c[qp]; //n+utheta
          //dUdt = 1.5*dUdt - .5*dUqp_old; //D2f^{n+1}/dt = 1.5D1f^{n+1}/dt-.5(U^{n}-U^{n-1})/dt
        }
        else if (is_bdf3 && time_step > 1)
        {
          Uqp_m1 = u_coefs_c_om1_trans * phi_c[qp];  //cout << u_coefs_c_om1_trans << endl;
          Uqp_m2 = u_coefs_c_om2_trans * phi_c[qp];  //cout << u_coefs_c_om2_trans << endl << endl;
          dUdt   = 11./6.*dUdt - 7./6.*Uqp_old/dt + 3./2.*Uqp_m1/dt - 1./3.*Uqp_m2/dt;
          //dUqp_old   = du_coefs_c_old_trans  * phi_c[qp];
          //dUqp_vold  = du_coefs_c_vold_trans * phi_c[qp];
          //cout << dUqp_vold.transpose() << "  " << (Uqp_m1 - Uqp_m2).transpose()/dt << " ### ";
          //cout << dUqp_old.transpose()  << "  " << (Uqp_old- Uqp_m1).transpose()/dt << " ### ";
          //for (int j = 0; j < nodes_per_cell; j++){cout << cell->getNodeId(j) << " ";} cout << endl;
          //idd++;
          //dUdt = 11./6.*dUdt - 7./6.*dUqp_old + 1./3.*dUqp_vold;  //D3f^{n+1}/dt = 11/6 D1f^{n+1}/dt
        }                                                         //      -7/6(U^{n}-U^{n-1})/dt + 1/3 (U^{n-1}-U^{n-2})/dt

        //dxU, dUdt, Uconv_qp correction (adding solid contribution)
        //dxZ = z_coefs_c_mid_trans.block(0,0,2,nodes_per_cell) *  dLphi_c[qp] * invF_c_mid;
        Zqp = Vector::Zero(dim);

        if (SVI && false){
          if (is_bdf2 && time_step > 0){
            Uqp_m1 = u_coefs_c_om1c_trans * phi_c[qp];
          }
          else if (is_bdf3 && time_step > 1){
            //u_coefs_c_om1 = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
            //VecGetValues(Vec_uzp_m1,  mapU_t.size(), mapU_t.data(), u_coefs_c_om1.data()); // bdf2,bdf3
            //u_coefs_c_om1_trans = u_coefs_c_om1.transpose();
            Uqp_m1 = u_coefs_c_om1c_trans * phi_c[qp];
            //u_coefs_c_om2 = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
            //VecGetValues(Vec_uzp_m2,  mapU_t.size(), mapU_t.data(), u_coefs_c_om2.data());
            //u_coefs_c_om2_trans = u_coefs_c_om2.transpose();
            Uqp_m2 = u_coefs_c_om2c_trans * phi_c[qp];
          }
          for (int J = 0; J < nodes_per_cell; J++){
            if (SV_c[J]){
              //mesh->getNodePtr(cell_nodes(j)/*cell->getNodeId(J)*/)->getCoord(XIp.data(),dim);
              XJp_old = x_coefs_c_old_trans.col(J);
              XJp_new = x_coefs_c_new_trans.col(J);
              XJp     = x_coefs_c_mid_trans.col(J);
              //for dUdt
              z_coefs_c_auo.col(J) = SolidVel(XJp_old,XG_0[SV_c[J]-1],z_coefs_c_old_trans.col(J),dim);
              z_coefs_c_aun.col(J) = SolidVel(XJp_new,XG_1[SV_c[J]-1],z_coefs_c_new_trans.col(J),dim);
              //for dxU
              uz_coefs_c.col(J)    = SolidVel(XJp,XG_mid[SV_c[J]-1],z_coefs_c_mid_trans.col(J),dim);
            }
          }

          dxZ      = (uz_coefs_c + vs_coefs_c_mid_trans) * dLphi_c[qp] * invF_c_mid;  //cout << dxZ << endl;
          Zqp_new  = (z_coefs_c_aun + vs_coefs_c_new_trans) * phi_c[qp];              //cout << Zqp_new << endl;
          Zqp_old  = (z_coefs_c_auo + vs_coefs_c_old_trans) * phi_c[qp];              //cout << Zqp_old << endl;
          Zqp      = (uz_coefs_c + vs_coefs_c_mid_trans) * phi_c[qp];                 //cout << Zqp << endl;

          dxU      += dxZ;                    //cout << dxU << endl;
          Uconv_qp += Zqp;
          dUdt     += (Zqp_new-Zqp_old)/dt;
          if (is_bdf2 && time_step > 0){
            dUdt += 1./2.*Zqp_new/dt - 1.*Zqp_old/dt + 1./2.*Uqp_m1/dt;
          }
          else if (is_bdf3 && time_step > 1){
            dUdt += 5./6.*Zqp_new/dt - 2.*Zqp_old/dt + 3./2.*Uqp_m1/dt - 1./3.*Uqp_m2/dt;
          }
        }

        //Volumetric force (force per unit of volume)//////////////////////////////////////////////////
        force_at_mid = force(Xqp,current_time+utheta*dt,tag);
        //////////////////////////////////////////////////

        //quadrature weight//////////////////////////////////////////////////
        weight = quadr_cell->weight(qp);
        JxW_mid = J_mid*weight;
        if (is_axis){
          JxW_mid = JxW_mid*2.0*pi*Xqp(0);
        }
        //////////////////////////////////////////////////

        //inverted and degenerated element test//////////////////////////////////////////////////
        if (J_mid < 1.e-20)
        {
          FEP_PRAGMA_OMP(critical)
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
            throw;
          }
        }
        //////////////////////////////////////////////////

        // ---------------- //////////////////////////////////////////////////
        //
        //  RESIDUAL AND JACOBIAN MATRIX
        //
        //  --------------- //////////////////////////////////////////////////

        for (int i = 0; i < n_dofs_u_per_cell/dim; ++i)
        {
          for (int c = 0; c < dim; ++c)
          {
            FUloc(i*dim + c) += JxW_mid*(  //Momentum residual
                   rho*(unsteady*dUdt(c) + has_convec*Uconv_qp.dot(dxU.row(c)))*phi_c[qp][i]  //aceleração
                  +visc*dxphi_c.row(i).dot(dxU.row(c) + dxU.col(c).transpose())  //rigidez  //transpose() here is to add 2 row-vectors
                  -force_at_mid(c)*phi_c[qp][i]  //força
                  -Pqp_new*dxphi_c(i,c) );        //pressão
            if (is_axis && c == 0){
              FUloc(i*dim + c) += JxW_mid*(  //Momentum residual
                     2.0*visc*phi_c[qp][i]*Uqp(c)/(Xqp(c)*Xqp(c))  //rigidez
                    -Pqp_new*phi_c[qp][i]/(Xqp(c)) );        //pressão
            }
            for (int j = 0; j < n_dofs_u_per_cell/dim; ++j)
            {
              for (int d = 0; d < dim; ++d)
              {
                delta_cd = c==d;
                Aloc(i*dim + c, j*dim + d) += JxW_mid*(
                      ddt_factor*unsteady*delta_cd*rho*phi_c[qp][i]*phi_c[qp][j]/dt + //time derivative
                      has_convec*phi_c[qp][i]*utheta*rho*( delta_cd*Uconv_qp.dot(dxphi_c.row(j)) + dxU(c,d)*phi_c[qp][j] ) + //advecção
                      utheta*visc*( delta_cd*dxphi_c.row(i).dot(dxphi_c.row(j)) + dxphi_c(i,d)*dxphi_c(j,c)) ); //rigidez
                if (is_axis && c == 0){
                  Aloc(i*dim + c, j*dim + d) += JxW_mid*(
                        utheta*2.0*visc*delta_cd*phi_c[qp][i]*phi_c[qp][j]/(Xqp(c)*Xqp(c)) ); //rigidez
                }
              }
            }
            for (int j = 0; j < n_dofs_p_per_cell; ++j)
            {
              Gloc(i*dim + c,j) -= JxW_mid * psi_c[qp][j]* dxphi_c(i,c);            //pressure block
              Dloc(j,i*dim + c) -= utheta*JxW_mid * psi_c[qp][j]*  dxphi_c(i,c);    //transpose of the pressure block
              if (is_axis && c == 0){
                Gloc(i*dim + c,j) -= JxW_mid * psi_c[qp][j] * phi_c[qp][i]/Xqp(c);    //pressure block
                Dloc(j,i*dim + c) -= utheta*JxW_mid * psi_c[qp][j] * phi_c[qp][i]/Xqp(c);  //transpose of the pressure block
              }
            }

          }
        }

        for (int i = 0; i < n_dofs_p_per_cell; ++i){
          FPloc(i) -= JxW_mid* dxU.trace()*psi_c[qp][i]; //mass conservation residual
          if (is_axis){
            FPloc(i) -= JxW_mid*psi_c[qp][i]*Uqp(0)/Xqp(0); //mass conservation residual
          }
        }

        // ---------------- //////////////////////////////////////////////////
        //
        //  STABILIZATION
        //
        //  --------------- //////////////////////////////////////////////////

        if(behaviors & BH_GLS)
        {
          //int st_met = 0;  //-1 for Douglas and Wang, +1 for Hughes and Franca
          //int st_vis = 0; //1 for including visc term in the residual
          bool test_st = true;
          double divu = 0;
          //Residual
          Res = rho*(unsteady*dUdt + has_convec*dxU*Uconv_qp) + dxP_new - force_at_mid;
          if (is_axis && test_st){//complementing the cylindrical laplacian
            Res(0) += -st_vis*utheta*visc*( dxU(0,0)/Xqp(0) - Uqp(0)/(Xqp(0)*Xqp(0)) );
            Res(1) += -st_vis*utheta*visc*( dxU(1,0)/Xqp(0)                          );
          }

          for (int i = 0; i < n_dofs_u_per_cell/dim; i++)
          {
            //supg term tauk
            vec  = JxW_mid*(has_convec)*tauk*rho*Uconv_qp.dot(dxphi_c.row(i))*Res;
            //divergence term delk
            divu = dxU.trace();
            if (is_axis){
              divu += Uqp(0)/Xqp(0);
            }
            vec += JxW_mid*delk*divu*dxphi_c.row(i).transpose();
            if (is_axis && test_st){
              //supg term tauk
              vec(0) += st_met*JxW_mid*tauk*utheta*Res(0)*visc*( dxphi_c(i,0)/Xqp(0) - phi_c[qp][i]/(Xqp(0)*Xqp(0)) );
              vec(1) += st_met*JxW_mid*tauk*utheta*Res(1)*visc*( dxphi_c(i,0)/Xqp(0)                                );
              //divergence term delk
              vec(0) += JxW_mid*delk*divu*phi_c[qp][i]/Xqp(0);
            }
            for (int c = 0; c < dim; c++)
              FUloc(i*dim + c) += vec(c);
          }

          for (int i = 0; i < n_dofs_p_per_cell; i++)
            FPloc(i) -= JxW_mid*tauk*dxpsi_c.row(i).dot(Res);

          //Gradient of Residual
          for (int j = 0; j < n_dofs_u_per_cell/dim; j++)
          {
            dResdu = unsteady*ddt_factor*rho*phi_c[qp][j]/dt*Id + has_convec*utheta*rho*( phi_c[qp][j]*dxU + Uconv_qp.dot(dxphi_c.row(j))*Id );
            if (is_axis && test_st){
              dResdu(0,0) += -st_vis*utheta*visc*( dxphi_c(j,0)/Xqp(0) - phi_c[qp][j]/(Xqp(0)*Xqp(0)) );
              dResdu(1,1) += -st_vis*utheta*visc*( dxphi_c(j,0)/Xqp(0)                                );
            }

            for (int i = 0; i < n_dofs_u_per_cell/dim; i++)
            {
              //supg term tauk
              Ten  = JxW_mid*(has_convec)*tauk*( utheta*rho*phi_c[qp][j]*Res*dxphi_c.row(i) + rho*Uconv_qp.dot(dxphi_c.row(i))*dResdu );
              // divergence term
              Ten += JxW_mid*delk*utheta*dxphi_c.row(i).transpose()*dxphi_c.row(j);
              if (is_axis && test_st){
                //Ten.row(0) += st_met*JxW_mid*tauk*utheta*visc*( dxphi_c(i,0)/Xqp(0) - phi_c[qp][i]/(Xqp(0)*Xqp(0)) )*dResdu.row(0);
                //Ten.row(1) += st_met*JxW_mid*tauk*utheta*visc*( dxphi_c(i,0)/Xqp(0)                                )*dResdu.row(1);
                Ten(0,0) += st_met*JxW_mid*tauk*utheta*visc*( dxphi_c(i,0)/Xqp(0) - phi_c[qp][i]/(Xqp(0)*Xqp(0)) )*dResdu(0,0);
                Ten(1,1) += st_met*JxW_mid*tauk*utheta*visc*( dxphi_c(i,0)/Xqp(0)                                )*dResdu(1,1);
              }
              for (int c = 0; c < dim; ++c)
                for (int d = 0; d < dim; ++d)
                  Aloc(i*dim + c, j*dim + d) += Ten(c,d);
            }//end for i

            for (int i = 0; i < n_dofs_p_per_cell; i++)
            {
              vec = -JxW_mid*tauk*dResdu.transpose()*dxpsi_c.row(i).transpose();
              for (int d = 0; d < dim; d++)
                Dloc(i, j*dim + d) += vec(d);

              vec = JxW_mid*tauk*(has_convec)*rho*Uconv_qp.dot(dxphi_c.row(j))*dxpsi_c.row(i).transpose();
              if (is_axis && test_st){
                vec(0) += st_met*JxW_mid*tauk*visc*dxpsi_c(i,0)*( dxphi_c(j,0)/Xqp(0) - phi_c[qp][j]/(Xqp(0)*Xqp(0)) );
                vec(1) += st_met*JxW_mid*tauk*visc*dxpsi_c(i,1)*( dxphi_c(j,0)/Xqp(0)                                );
              }
              for (int d = 0; d < dim; d++)
                Gloc(j*dim + d,i) += vec(d);
            }//end for i
          }//end for j

          for (int i = 0; i < n_dofs_p_per_cell; ++i)
            for (int j = 0; j < n_dofs_p_per_cell; ++j)
              Eloc(i,j) -= JxW_mid*tauk*dxpsi_c.row(i).dot(dxpsi_c.row(j));

        }//end if behaviors
        areas += JxW_mid;
      }
      ////////////////////////////////////////////////// ENDING QUADRATURE //////////////////////////////////////////////////

//cout << "\n" << FUloc << endl; cout << "\n" << Aloc << endl; cout << "\n" << Gloc << endl; cout << "\n" << Dloc << endl;

      // Projection - to force Dirichlet conditions: solid motion, non-penetration, pure Dirichlet, respect //////////////////////////////////////////////////
      if (SVI || VSF || PDN || FTN){
        VectorXd    FUloc_copy(n_dofs_u_per_cell);
        MatrixXd    Aloc_copy(n_dofs_u_per_cell, n_dofs_u_per_cell);
        MatrixXd    Gloc_copy(n_dofs_u_per_cell, n_dofs_p_per_cell);

        FUloc_copy = FUloc;
        Aloc_copy  = Aloc;
        Gloc_copy  = Gloc;
        //Asloc.setZero();
        //Dsloc.setZero();

        //mesh->getCellNodesId(&*cell, cell_nodes.data());  //cout << cell_nodes.transpose() << endl;
        getProjectorMatrix(Prj, nodes_per_cell, cell_nodes.data(), Vec_x_1, current_time+dt, *this);
        FUloc = Prj*FUloc;  //if feature_proj case, vanishes the cartesian component contribution you want to impose (non penetration or slip velocity)
        Aloc = Prj*Aloc; //Prj*Aloc*Prj;//if feature_proj case, vanishes the line and column of the component you want to impose
        Gloc = Prj*Gloc;
        //Dloc = Dloc*Prj;
        //Z1loc = Prj*Z1loc;
        //Z2loc = Z2loc*Prj;
//cout << "\n" << FUloc << endl; cout << "\n" << Aloc << endl; cout << "\n" << Gloc << endl; cout << "\n" << Dloc << endl;

        //impose non-penetration BC at body's surface nodes
        VectorXd    FUloc_vs(VectorXd::Zero(n_dofs_u_per_cell));
        //VectorXi    mapZ_c_vs = -VectorXi::Ones(nodes_per_cell*LZ);
        MatrixXd    Hq_vs(MatrixXd::Zero(n_dofs_u_per_cell,n_dofs_z_per_cell));
        double      betaPrj = 1.0;

        //z_coefs_c_new = MatrixXd::Zero(n_dofs_z_per_cell/LZ,LZ);

        for (int i = 0; i < n_dofs_u_per_cell/dim; ++i){
          int K = SV_c[i] + VS_c[i];
          if (K){
            XIp    = x_coefs_c_new_trans.col(i); //ref point Xp, old, mid, or new
            XIg    = XG_mid[K-1];          //mass center, mid, _0, "new"
            RotfI  = SolidVel(XIp, XIg, z_coefs_c_new_trans.col(i), dim);

            for (int c = 0; c < dim; ++c){
              FUloc_vs(i*dim + c) = u_coefs_c_mid_trans(c,i) - RotfI(c);// - is_unksv*vs_coefs_c_mid_trans(c,i);
              if (SV_c[i]){
                FUloc_vs(i*dim + c) += - vs_coefs_c_mid_trans(c,i);
              }
              for (int D = 0; D < LZ; D++){
                RotfJ  = SolidVel(XIp, XIg, IdZ.col(D), dim);
                Hq_vs(i*dim+c,i*LZ+D) = RotfJ(c);  //builds the elemental matrix H of solid generator movement
              }
            }
          }
        }

        FUloc += betaPrj*(Id_vs - Prj)*FUloc_vs;
        Aloc  += betaPrj*(Id_vs - Prj);
        Z1loc = -betaPrj*(Id_vs - Prj)*Hq_vs;

        FZloc  = Hq_vs.transpose()*(Id_vs - Prj)*FUloc_copy;
        Z2loc  = Hq_vs.transpose()*(Id_vs - Prj)*Aloc_copy;
        Z4loc  = Hq_vs.transpose()*(Id_vs - Prj)*Gloc_copy;

        //if (is_unksv && (SVI || VSF)){
        //  Asloc = -betaPrj*(Id_vs - Prj);
        //}

        if (true /*is_axis && is_sfip*/){// eliminating specific solid DOFS
          MatrixXd PrjDOFS(n_dofs_z_per_cell,n_dofs_z_per_cell);
          MatrixXd Id_LZ(MatrixXd::Identity(n_dofs_z_per_cell,n_dofs_z_per_cell));
          VectorXi s_DOFS(LZ); s_DOFS = DOFS_elimination(LZ); //<< 0, 1, 0;
          if (is_sflp){
            MatrixXd PrjDOFSlz(LZ,LZ);
            PrjDOFS.setIdentity();
            for (int i = 0; i < n_dofs_u_per_cell/dim; ++i){
              int K = SV_c[i] + VS_c[i];
              if (K == 1)
                continue;

              getProjectorDOFS(PrjDOFSlz, 1, s_DOFS.data(), *this);
              PrjDOFS.block(i*LZ,i*LZ,LZ,LZ) = PrjDOFSlz;
            }
            FZloc = PrjDOFS*FZloc;
            Z2loc = PrjDOFS*Z2loc;
            Z4loc = PrjDOFS*Z4loc;
            Z3loc = Id_LZ - PrjDOFS;
          }
          else{
            getProjectorDOFS(PrjDOFS, n_dofs_u_per_cell/dim, s_DOFS.data(), *this);
            FZloc = PrjDOFS*FZloc;
            Z2loc = PrjDOFS*Z2loc;
            Z4loc = PrjDOFS*Z4loc;
            Z3loc.setZero(); //= Id_LZ - PrjDOFS;
          }

        }
      }
      //////////////////////////////////////////////////

      //////////////////////////////////////////////////
      if (force_pressure)
      {
        for (int i = 0; i < mapP_c.size(); ++i)
        {
          if (mapP_c(i) == null_space_press_dof)
          {
            Gloc.col(i).setZero();
            Dloc.row(i).setZero();
            FPloc(i) = 0;
            Eloc.col(i).setZero();
            Eloc.row(i).setZero();
            break;
          }
        }
      }
      //////////////////////////////////////////////////

      // ---------------- //////////////////////////////////////////////////
      //
      //  MATRIX ASSEMBLY
      //
      //  --------------- //////////////////////////////////////////////////
#ifdef FEP_HAS_OPENMP
      FEP_PRAGMA_OMP(critical)
#endif
      {
        VecSetValues(Vec_fun_fs, mapU_c.size(), mapU_c.data(), FUloc.data(), ADD_VALUES);
        VecSetValues(Vec_fun_fs, mapP_c.size(), mapP_c.data(), FPloc.data(), ADD_VALUES);
        if (is_sfip){
          VecSetValues(Vec_fun_fs, mapZ_c.size(), mapZ_c.data(), FZloc.data(), ADD_VALUES);
        }
        MatSetValues(*JJ, mapU_c.size(), mapU_c.data(), mapU_c.size(), mapU_c.data(), Aloc.data(), ADD_VALUES);
        MatSetValues(*JJ, mapU_c.size(), mapU_c.data(), mapP_c.size(), mapP_c.data(), Gloc.data(), ADD_VALUES);
        MatSetValues(*JJ, mapP_c.size(), mapP_c.data(), mapU_c.size(), mapU_c.data(), Dloc.data(), ADD_VALUES);
        MatSetValues(*JJ, mapP_c.size(), mapP_c.data(), mapP_c.size(), mapP_c.data(), Eloc.data(), ADD_VALUES);
        if (is_sfip){
          MatSetValues(*JJ, mapU_c.size(), mapU_c.data(), mapZ_c.size(), mapZ_c.data(), Z1loc.data(), ADD_VALUES);
          MatSetValues(*JJ, mapZ_c.size(), mapZ_c.data(), mapU_c.size(), mapU_c.data(), Z2loc.data(), ADD_VALUES);
          MatSetValues(*JJ, mapZ_c.size(), mapZ_c.data(), mapZ_c.size(), mapZ_c.data(), Z3loc.data(), ADD_VALUES);
          MatSetValues(*JJ, mapZ_c.size(), mapZ_c.data(), mapP_c.size(), mapP_c.data(), Z4loc.data(), ADD_VALUES);
          MatSetValues(*JJ, mapP_c.size(), mapP_c.data(), mapZ_c.size(), mapZ_c.data(), Z5loc.data(), ADD_VALUES);
          //if (is_unksv){
          //  MatSetValues(*JJ, mapU_c.size(), mapU_c.data(), mapUs_c.size(), mapUs_c.data(), Asloc.data(), ADD_VALUES);
          //  MatSetValues(*JJ, mapUs_t.size(), mapUs_t.data(), mapUs_t.size(), mapUs_t.data(), Id_vs.data(), ADD_VALUES);//imposing zero uslip for fluid nodes
          //}
        }
      }
    }  //end for cell
    //if (is_axis){cout << "Area = " << areas << ", Area dif = "<< areas-(300.0*300.0*pi*300-4.0*pi/3.0) << endl;} //Assembly(Vec_fun_fs);  Assembly(*JJ);
    //else        {cout << "Area = " << areas << ", Area dif = "<< areas-(10*10-pi/2.0) << endl;}
    //View(Vec_fun_fs, "matrizes/rhs.m","res"); View(*JJ,"matrizes/jacob.m","Jac");
  }
  // end LOOP NAS CÉLULAS Parallel (uncomment it) //////////////////////////////////////////////////


  // LOOP FOR SOLID-ONLY CONTRIBUTION //////////////////////////////////////////////////
  if (is_sfip /*&& unsteady*/)
  {
    VectorXd   FZsloc = VectorXd::Zero(LZ);
    VectorXi   mapZ_s(LZ), mapZ_J(LZ);
    VectorXd   z_coefs_old(LZ), z_coefs_new(LZ), z_coefs_om1(LZ), z_coefs_om2(LZ);
    VectorXd   z_coefs_mid(LZ), z_coefs_olJ(LZ), z_coefs_neJ(LZ), z_coefs_miJ(LZ), z_coefs_miK(LZ);
    Vector     dZdt(LZ);
    Vector     Grav(LZ), Fpp(LZ), Fpw(LZ), Fv(LZ);
    TensorZ    Z3sloc = TensorZ::Zero(LZ,LZ), dFpp(LZ,LZ), dFpw(LZ,LZ);
    TensorZ    MI = TensorZ::Zero(LZ,LZ);
    double     ddt_factor, dJK;
    bool       deltaDi, deltaLK, deltaLJ;
    Vector3d   eJK;
    double     zet = 1.0e-0, ep = 5.0e-1, epw = 1e-5; //ep/10.0;
    double     gap, R, visc;

    if (is_bdf2 && time_step > 0)
      ddt_factor = 1.5;
    else
    if (is_bdf3 && time_step > 1)
      ddt_factor = 11./6.;
    else
      ddt_factor = 1.;

    visc = muu(0);

    for (int K = 0; K < n_solids; K++){

      Fpp  = Vector::Zero(LZ);     Fpw  = Vector::Zero(LZ);
      dFpp = TensorZ::Zero(LZ,LZ); dFpw = TensorZ::Zero(LZ,LZ);
      Grav = gravity(XG_mid[K], dim); Fv = Grav;

      for (int C = 0; C < LZ; C++){
        mapZ_s(C) = n_unknowns_u + n_unknowns_p + LZ*K + C;
      }  //cout << mapZ_s << endl;
      VecGetValues(Vec_uzp_0,    mapZ_s.size(), mapZ_s.data(), z_coefs_old.data());  //cout << z_coefs_old.transpose() << endl;
      VecGetValues(Vec_uzp_k ,   mapZ_s.size(), mapZ_s.data(), z_coefs_new.data());  //cout << z_coefs_new.transpose() << endl;
      VecGetValues(Vec_uzp_m1,   mapZ_s.size(), mapZ_s.data(), z_coefs_om1.data()); // bdf2,bdf3
      if (is_bdf3){
        VecGetValues(Vec_uzp_m2, mapZ_s.size(), mapZ_s.data(), z_coefs_om2.data()); // bdf2
      }

      z_coefs_mid = utheta*z_coefs_new + (1-utheta)*z_coefs_old;

      //Rep force Wang
#if (false)
      hme = zet;
      for (int L = 0; L < n_solids; L++){
        if (L != K){
          Fpp += force_pp(XG_mid[K], XG_mid[L], RV[K](0), RV[L](0),
                          ep, ep, hme);
        }
      }
      {
        Vector coor(dim);
        Vector3d   Xj;  int widp = 2*n_solids;
        mesh->getNodePtr(widp)->getCoord(coor.data(),dim);  //cout << coor.transpose() << "   ";
        Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_pw(XG_mid[K], Xj, RV[K](0), ep, ep*ep, hme);  //cout << Fpw.transpose() << endl;

        Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_pw(XG_mid[K], Xj, RV[K](0), ep, ep*ep, hme);  //cout << Fpw.transpose() << endl;

        mesh->getNodePtr(widp+2)->getCoord(coor.data(),dim);  //cout << coor.transpose() << endl;
        Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_pw(XG_mid[K], Xj, RV[K](0), ep, ep*ep, hme);  //cout << Fpw.transpose() << endl;

        Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_pw(XG_mid[K], Xj, RV[K](0), ep, ep*ep, hme);  //cout << Fpw.transpose() << endl;

        if ((Fpp.norm() != 0) || (Fpw.norm() != 0)){
          cout << K+1 << "   " << Fpp.transpose() << "   " << Fpw.transpose() << endl;
        }
      }
#endif
      //Rep force Luzia
#if (false)
      double INF = 1.0e5;
      MatrixXd ContP(MatrixXd::Zero(n_solids,n_solids)), ContW(MatrixXd::Zero(n_solids,5));
      bool RepF = proxTest(ContP, ContW, INF);
      if (RepF){
        //Point to Point
        for (int L = 0; L < n_solids; L++){
          ep = ContP(K,L);  zet = 0.92;//zet = 0.92;
          if ((L != K) && (ep < INF)){
            Fpp += force_ppl(XG_mid[K], XG_mid[L], ep, zet);
          }
        }
        Vector coor(dim);
        Vector3d   Xj;  int widp = 2*n_solids;
        //Point to wall
        mesh->getNodePtr(widp)->getCoord(coor.data(),dim);  //left-inf corner
        ep = ContW(K,0);  zet = 0.92;
        if (ep < INF){
          Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
          Fpw += force_ppl(XG_mid[K], Xj, ep, zet);  //cout << Fpw.transpose() << endl;
        }
        ep = ContW(K,1);  zet = 0.92;
        if (ep < INF){
          Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
          Fpw += force_ppl(XG_mid[K], Xj, ep, zet);  //cout << Fpw.transpose() << endl;
        }
        mesh->getNodePtr(widp+2)->getCoord(coor.data(),dim);  //rigth-sup corner
        ep = ContW(K,2);  zet = 0.92;
        if (ep < INF){
          Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
          Fpw += force_ppl(XG_mid[K], Xj, ep, zet);  //cout << Fpw.transpose() << endl;
        }
        ep = ContW(K,3);  zet = 0.92;
        if (ep < INF){
          Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
          Fpw += force_ppl(XG_mid[K], Xj, ep, zet);  //cout << Fpw.transpose() << endl;
        }

        if ((Fpp.norm() != 0) || (Fpw.norm() != 0)){
          cout << K << "   " << Fpp.transpose() << "   " << Fpw.transpose() << endl;
          //cin.get();
        }
      }// end repulsion force
#endif
      //Rep force Glowinski
#if (false)
      for (int L = 0; L < n_solids; L++){
        if (L != K){
          //if (RepF){zet = ContP(K,L); ep = zet*zet;}
          Fpp += force_rga(XG_mid[K], XG_mid[L], RV[K](0), RV[L],
                           Grav, MV[K], ep, zet);
        }
      }
      {
        Vector coor(dim);
        Vector3d   Xj;  int widp = 2*n_solids; //6*n_solids;
        mesh->getNodePtr(widp)->getCoord(coor.data(),dim);  //cout << coor.transpose() << "   ";
        Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rga(XG_mid[K], Xj, RV[K](0), RV[K](0), Grav, MV[K], ep, zet);

        Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rga(XG_mid[K], Xj, RV[K](0), RV[K](0), Grav, MV[K], ep, zet);

        mesh->getNodePtr(widp+2)->getCoord(coor.data(),dim);  //cout << coor.transpose() << endl;
        Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rga(XG_mid[K], Xj, RV[K](0), RV[K](0), Grav, MV[K], ep, zet);

        Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rga(XG_mid[K], Xj, RV[K](0), RV[K](0), Grav, MV[K], ep, zet);

        if ((Fpp.norm() != 0) || (Fpw.norm() != 0)){
          cout << K << "   " << Fpp.transpose() << "   " << Fpw.transpose() << endl;
        }
      }
#endif
      //Rep force Glowinski
#if (false)
      for (int L = 0; L < n_solids; L++){
        if (L != K){
          //if (RepF){zet = ContP(K,L); ep = zet*zet;}
          Fpp += force_rgc(XG_mid[K], XG_mid[L], RV[K](0), RV[L], ep, zet);
        }
      }
      { //This part is sensibly: the 3*n_solids part depends on the gmsh structure box corner creation
        Vector coor(dim);
        Vector3d   Xj;  int widp = 0; //6*n_solids;//0; //3*n_solids
        mesh->getNodePtr(widp)->getCoord(coor.data(),dim);  //cout << coor.transpose() << endl;
        Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rgc(XG_mid[K], Xj, RV[K](0), RV[K](0), epw, zet);

        Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rgc(XG_mid[K], Xj, RV[K](0), RV[K](0), epw, zet);

        mesh->getNodePtr(widp+2)->getCoord(coor.data(),dim);  //cout << coor.transpose() << endl;
        Xj << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rgc(XG_mid[K], Xj, RV[K](0), RV[K](0), epw, zet);

        Xj << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;   //cout << Xj.transpose() << endl;
        Fpw += force_rgc(XG_mid[K], Xj, RV[K](0), RV[K](0), epw, zet);

        if ((Fpp.norm() != 0) || (Fpw.norm() != 0)){
          cout << K << "   " << Fpp.transpose() << "   " << Fpw.transpose() << endl;
        }
      }
#endif
      //Rep force Buscaglia
#if (false)
      zet = 0.01;
      for (int J = 0; J < n_solids; J++){
        if (J != K){
          eJK = XG_mid[K]-XG_mid[J];
          dJK = eJK.norm();
          ep  = dJK-(RV[K](0)+RV[J]); //cout << ep << " ";
          if (ep <= zet){
            R = std::max(RV[K](0),RV[J]);
            //ep = zet;
            for (int C = 0; C < LZ; C++){
              mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*J + C;
            }
            VecGetValues(Vec_uzp_0,    mapZ_J.size(), mapZ_J.data(), z_coefs_olJ.data());  //cout << z_coefs_old.transpose() << endl;
            VecGetValues(Vec_uzp_k ,   mapZ_J.size(), mapZ_J.data(), z_coefs_neJ.data());  //cout << z_coefs_new.transpose() << endl;
            z_coefs_miJ = utheta*z_coefs_neJ + (1-utheta)*z_coefs_olJ;
            gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep))*(z_coefs_miJ-z_coefs_mid).norm();
            Fpp.head(3) += gap*eJK/dJK;
            cout << K+1 << " " << J+1 << " - ";
          }
        }
      }
      for (int L = 0; L < n_solids; L++){
        deltaLK = L==K;
        for (int J = 0; J < n_solids; J++){
          deltaLJ = L==J;
          if (J != K){
            eJK = XG_mid[K]-XG_mid[J];
            dJK = eJK.norm();
            ep  = dJK-(RV[K](0)+RV[J]);
            if (ep <= zet){
              R = std::max(RV[K](0),RV[J]);
              //ep = zet;
              for (int C = 0; C < LZ; C++){
                mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*J + C;
              }
              VecGetValues(Vec_uzp_0,    mapZ_J.size(), mapZ_J.data(), z_coefs_olJ.data());  //cout << z_coefs_old.transpose() << endl;
              VecGetValues(Vec_uzp_k ,   mapZ_J.size(), mapZ_J.data(), z_coefs_neJ.data());  //cout << z_coefs_new.transpose() << endl;
              z_coefs_miJ = utheta*z_coefs_neJ + (1-utheta)*z_coefs_olJ;
              gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep));//*(z_coefs_miJ-z_coefs_mid).norm();
              for (int C = 0; C < dim; C++){
                for (int D = 0; D < dim; D++){
                  for (int i = 0; i < dim; i++){
                    deltaDi = D==i;
                    dFpp(C,D) -= utheta*gap*deltaDi*(deltaLK-deltaLJ)/(z_coefs_miJ-z_coefs_mid).norm() * eJK(C)/dJK;
                  }
                }
              }
            }
          }
        }
        for (int C = 0; C < LZ; C++){
          mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*L + C;
        }
        MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_J.size(), mapZ_J.data(), dFpp.data(), ADD_VALUES);
      }

      { //This part is sensibly: the 3*n_solids part depends on the gmsh structure box corner creation
        zet = 0.02;
        Vector   coor(dim), coor1(dim), coor2(dim);
        Vector3d Xkaux; int widp = 2*n_solids;// 6*n_solids;//9; //0 2*n_solids; //choose the left-inferior corner as reference
        // bottom wall
        mesh->getNodePtr(widp)->getCoord(coor.data(),dim);  //cout << coor.transpose() << "   ";
        Xkaux << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;
        eJK = XG_mid[K]-Xkaux;
        dJK = eJK.norm();
        ep  = dJK-(2*RV[K](0));
        if (ep <= zet){
          R = RV[K](0);
          //ep = zet;
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep))*2*(z_coefs_mid.head(dim)).norm();
          Fpw.head(3) += gap * eJK/dJK; //(-z_coefs_mid.head(dim)/z_coefs_mid.head(dim).norm())
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep));//*(z_coefs_mid.head(dim)).norm();
          for (int L = 0; L < n_solids; L++){
            deltaLK = L==K;
            for (int C = 0; C < dim; C++){
              for (int D = 0; D < dim; D++){
                for (int i = 0; i < dim; i++){
                  deltaDi = D==i;
                  dFpw(C,D) -= 2*utheta*gap*deltaDi*(deltaLK)/(z_coefs_mid).norm() * eJK(C)/dJK; //-z_coefs_mid(C)/z_coefs_mid.head(dim).norm()
                }
              }
            }
            for (int C = 0; C < LZ; C++){
              mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*L + C;
            }
            MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_J.size(), mapZ_J.data(), dFpp.data(), ADD_VALUES);
          }
          cout << "bottom  ";
        }
/*        double dism = 10;
        point_iterator point1 = mesh->pointBegin();
        point_iterator point1_end = mesh->pointEnd();
        point_iterator point2, point2_end;

        for (; point1 != point1_end; ++point1)
        {
          int tag1 = point1->getTag();
          point2 = mesh->pointBegin();
          point2_end = mesh->pointEnd();
          for (; point2 != point2_end; ++point2)
          {
            int const tag2 = point2->getTag();
            if (!(is_in(tag1,flusoli_tags) && tag2 == 10))
              continue;
            point1->getCoord(coor1.data(),dim);
            point2->getCoord(coor2.data(),dim);
            if (dism > (coor1-coor2).norm())
              dism = (coor1-coor2).norm();
          }
        }*/
        //zet = 0.01;
        // left wall
        Xkaux << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;
        eJK = XG_mid[K]-Xkaux;
        dJK = eJK.norm();//dism;
        ep  = dJK-(2*RV[K](0));//dism;
        //cout << dism << "  " << eJK.transpose()/dism << "  ";
        if (ep <= zet){
          R = RV[K](0);
          //ep = zet;
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep))*2*(z_coefs_mid).norm();
          Fpw.head(3) += gap*eJK/eJK.norm();//dJK;
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep));//*(z_coefs_mid).norm();
          for (int L = 0; L < n_solids; L++){
            deltaLK = L==K;
            for (int C = 0; C < dim; C++){
              for (int D = 0; D < dim; D++){
                for (int i = 0; i < dim; i++){
                  deltaDi = D==i;
                  dFpw(C,D) -= 2*utheta*gap*deltaDi*(deltaLK)/(z_coefs_mid).norm() * eJK(C)/eJK.norm();//dJK;
                }
              }
            }
            for (int C = 0; C < LZ; C++){
              mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*L + C;
            }
            MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_J.size(), mapZ_J.data(), dFpp.data(), ADD_VALUES);
          }
          cout << "left  ";
        }
        // right wall
        widp = widp + 2;
        mesh->getNodePtr(widp)->getCoord(coor.data(),dim);  //cout << coor.transpose() << "   ";
        Xkaux << 2*coor[0]-XG_mid[K](0), XG_mid[K](1), 0.0;
        eJK = XG_mid[K]-Xkaux;
        dJK = eJK.norm();
        ep  = dJK-(2*RV[K](0));
        if (ep <= zet){
          R = RV[K](0);
          //ep = zet;
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep))*2*(z_coefs_mid).norm();
          Fpw.head(3) += gap*eJK/dJK;
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep));//*(z_coefs_mid).norm();
          for (int L = 0; L < n_solids; L++){
            deltaLK = L==K;
            for (int C = 0; C < dim; C++){
              for (int D = 0; D < dim; D++){
                for (int i = 0; i < dim; i++){
                  deltaDi = D==i;
                  dFpw(C,D) -= 2*utheta*gap*deltaDi*(deltaLK)/(z_coefs_mid).norm() * eJK(C)/dJK;
                }
              }
            }
            for (int C = 0; C < LZ; C++){
              mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*L + C;
            }
            MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_J.size(), mapZ_J.data(), dFpp.data(), ADD_VALUES);
          }
          cout << "right  ";
        }
        //top wall
        Xkaux << XG_mid[K](0), 2*coor[1]-XG_mid[K](1), 0.0;
        eJK = XG_mid[K]-Xkaux;
        dJK = eJK.norm();
        ep  = dJK-(2*RV[K](0));
        if (false && ep <= zet){
          R = RV[K](0);
          //ep = zet;
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep))*2*(z_coefs_mid).norm();
          Fpw.head(3) += gap*eJK/dJK;
          gap = 8.0*visc*sqrt(R*R*R/(ep*ep*ep));//*(z_coefs_mid).norm();
          for (int L = 0; L < n_solids; L++){
            deltaLK = L==K;
            for (int C = 0; C < dim; C++){
              for (int D = 0; D < dim; D++){
                for (int i = 0; i < dim; i++){
                  deltaDi = D==i;
                  dFpw(C,D) -= 2*utheta*gap*deltaDi*(deltaLK)/(z_coefs_mid).norm() * eJK(C)/dJK;
                }
              }
            }
            for (int C = 0; C < LZ; C++){
              mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*L + C;
            }
            MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_J.size(), mapZ_J.data(), dFpp.data(), ADD_VALUES);
          }
          cout << "top  ";
        }

        if ((Fpp.norm() != 0) || (Fpw.norm() != 0)){
          cout << K+1 << "   " << Fpp.transpose() << "   " << Fpw.transpose() << endl;
        }
      }
#endif

      dZdt = (z_coefs_new - z_coefs_old)/dt;
      if (is_bdf2 && time_step > 0){
        dZdt = 3./2.*dZdt - 1./2.*z_coefs_old/dt + 1./2.*z_coefs_om1/dt;
      }
      else if (is_bdf3 && time_step > 1){
        dZdt = 11./6.*dZdt - 7./6.*z_coefs_old/dt + 3./2.*z_coefs_om1/dt - 1./3.*z_coefs_om2/dt;
      }
      MI = MI_tensor(MV[K],RV[K](0),dim,InTen[K]);
      FZsloc = (MI*dZdt - MV[K]*Grav)*unsteady - Fpp - Fpw - Fv;
      Z3sloc = ddt_factor*MI/dt * unsteady;
//#ifdef FEP_HAS_OPENMP
//      FEP_PRAGMA_OMP(critical)
//#endif
      {
        VecSetValues(Vec_fun_fs, mapZ_s.size(), mapZ_s.data(), FZsloc.data(), ADD_VALUES);
        MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_s.size(), mapZ_s.data(), Z3sloc.data(), ADD_VALUES);
      }

    }//end K
    //Assembly(Vec_fun_fs); Assembly(*JJ);
    //View(Vec_fun_fs, "matrizes/rhs.m","res"); View(*JJ,"matrizes/jacob.m","Jac");
  }//cout << endl;
  // end LOOP FOR SOLID-ONLY CONTRIBUTION //////////////////////////////////////////////////


  // LOOP FOR LINK PROBLEM CONTRIBUTION //////////////////////////////////////////////////
  if (is_sflp){
    VectorXi    mapL_l(n_links);
    VectorXd    l_coefs_old(n_links), l_coefs_new(n_links), l_coefs_mid(n_links);
    VectorXd    l_coefs_tmp(VectorXd::Ones(n_links));
    Vector      FZsloc = Vector::Zero(LZ);
    VectorXd    z_coefs_old(LZ), z_coefs_new(LZ), z_coefs_mid(LZ);
    VectorXd    z_coefs_old_ref(LZ), z_coefs_new_ref(LZ), z_coefs_mid_ref(LZ);
    VectorXi    mapZ_s(LZ), mapZ_J(LZ);
    TensorZ     Z3sloc = TensorZ::Zero(LZ,LZ);
    MatrixXd    Zlsloc(MatrixXd::Identity(n_links,n_links));
    Vector      Lsloc = Vector::Zero(LZ);
    VectorXd    z_coefs_tmp(VectorXd::Zero(LZ));



    for (int nl = 0; nl < n_links; nl++){
      mapL_l(nl) = n_unknowns_u + n_unknowns_p + n_unknowns_z + n_modes + nl;
    }
    VecGetValues(Vec_uzp_0, mapL_l.size(), mapL_l.data(), l_coefs_old.data());  //cout << z_coefs_old.transpose() << endl;
    VecGetValues(Vec_uzp_k, mapL_l.size(), mapL_l.data(), l_coefs_new.data());
    l_coefs_mid = utheta*l_coefs_new + (1-utheta)*l_coefs_old;
    MatSetValues(*JJ, mapL_l.size(), mapL_l.data(), mapL_l.size(), mapL_l.data(), Zlsloc.data(), INSERT_VALUES);

    for (int C = 0; C < LZ; C++){
      mapZ_s(C) = n_unknowns_u + n_unknowns_p + C;
    }
    VecGetValues(Vec_uzp_0, mapZ_s.size(), mapZ_s.data(), z_coefs_old_ref.data());  //cout << z_coefs_old.transpose() << endl;
    VecGetValues(Vec_uzp_k, mapZ_s.size(), mapZ_s.data(), z_coefs_new_ref.data());  //cout << z_coefs_new.transpose() << endl;
    z_coefs_mid_ref = utheta*z_coefs_new_ref + (1-utheta)*z_coefs_old_ref;

    Lsloc = -LinksVel(XG_mid[0], XG_mid[0], z_coefs_tmp, theta_1[0], l_coefs_tmp, 2, ebref[0], dim, LZ);  //cout << Lsloc.transpose() << endl;

    for (int K = 0; K < n_solids; K++){
      if (K == 0)
        continue;

      for (int C = 0; C < LZ; C++){
        mapZ_s(C) = n_unknowns_u + n_unknowns_p + LZ*K + C;
      }
      VecGetValues(Vec_uzp_0, mapZ_s.size(), mapZ_s.data(), z_coefs_old.data());  //cout << z_coefs_old.transpose() << endl;
      VecGetValues(Vec_uzp_k, mapZ_s.size(), mapZ_s.data(), z_coefs_new.data());  //cout << z_coefs_new.transpose() << endl;
      z_coefs_mid = utheta*z_coefs_new + (1-utheta)*z_coefs_old;

      FZsloc = z_coefs_mid-LinksVel(XG_mid[K], XG_mid[0], z_coefs_mid_ref, theta_1[0], l_coefs_new, K+1, ebref[0], dim, LZ);
      VecSetValues(Vec_fun_fs, mapZ_s.size(), mapZ_s.data(), FZsloc.data(), INSERT_VALUES);

      for (int nl = 0; nl < K; nl++){
        MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), 1, &mapL_l(nl), Lsloc.data(), INSERT_VALUES);
      }

      for (int L = 0; L < n_solids; L++){
        for (int C = 0; C < LZ; C++){
          mapZ_J(C) = n_unknowns_u + n_unknowns_p + LZ*L + C;
        }
        if (L == 0){
          Z3sloc = -TensorZ::Identity(LZ,LZ);
          Vector Xgd = XG_mid[K] - XG_mid[0];
          Z3sloc(0,LZ-1) = -(-Xgd(1));
          Z3sloc(1,LZ-1) = -Xgd(0);
        }
        else if (L == K){
          Z3sloc = TensorZ::Identity(LZ,LZ);
        }
        else{
          Z3sloc = TensorZ::Zero(LZ,LZ);
        }
        MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_J.size(), mapZ_J.data(), Z3sloc.data(), INSERT_VALUES);
      }
    }
  }
  // LOOP FOR LINK PROBLEM CONTRIBUTION //////////////////////////////////////////////////


  // LOOP NAS FACES DO CONTORNO (Neum, Interf, Sol, NeumBody) //////////////////////////////////////////////////
  //~ FEP_PRAGMA_OMP(parallel default(none) shared(Vec_uzp_k,Vec_fun_fs,cout))
  {
    int                 tag, sid, nod_id, nod_sv, nodsum, pts_id[15], is_slipvel, is_fsi;//tag_p
    bool                is_neumann, is_surface, is_solid;

    std::vector<int>    SV_f(nodes_per_facet,0);   //maximum nodes in slip visited saving the solid tag
    //bool                SVI;// = false;              //slip velocity interaction, body pure Dirichlet node detected
    std::vector<int>    VS_f(nodes_per_facet,0);   //maximum nodes in visited solid saving the solid tag
    //bool                VSF;// = false;              //visited solid-fluid, body mixed Neumann node detected

    VectorXi            mapU_f(n_dofs_u_per_facet), mapU_t(n_dofs_u_per_facet);
    VectorXi            mapP_f(n_dofs_p_per_facet);
    VectorXi            mapM_f(dim*nodes_per_facet), mapS_f(dim*nodes_per_facet);
    VectorXi            mapZ_f(nodes_per_facet*LZ), mapZ_s(LZ);
    VectorXi            mapUs_f(n_dofs_u_per_facet);

    MatrixXd            u_coefs_f_mid_trans(dim, n_dofs_u_per_facet/dim);  // n+utheta
    MatrixXd            u_coefs_f_old(n_dofs_u_per_facet/dim, dim);        // n
    MatrixXd            u_coefs_f_old_trans(dim,n_dofs_u_per_facet/dim);   // n
    MatrixXd            u_coefs_f_new(n_dofs_u_per_facet/dim, dim);        // n+1
    MatrixXd            u_coefs_f_new_trans(dim,n_dofs_u_per_facet/dim);   // n+1

    MatrixXd            x_coefs_f_mid_trans(dim, n_dofs_v_per_facet/dim); // n+utheta
    MatrixXd            x_coefs_f_old(n_dofs_v_per_facet/dim, dim);       // n
    MatrixXd            x_coefs_f_old_trans(dim, n_dofs_v_per_facet/dim); // n
    MatrixXd            x_coefs_f_new(n_dofs_v_per_facet/dim, dim);       // n+1
    MatrixXd            x_coefs_f_new_trans(dim, n_dofs_v_per_facet/dim); // n+1

    MatrixXd            z_coefs_f_mid_trans(LZ,n_dofs_z_per_facet/LZ);   // n+1
    MatrixXd            z_coefs_f_new(n_dofs_z_per_facet/LZ, LZ);        // n+1
    MatrixXd            z_coefs_f_new_trans(LZ,n_dofs_z_per_facet/LZ);   // n+1

    MatrixXd            noi_coefs_f_new(n_dofs_v_per_facet/dim, dim);  // normal interpolada em n+1
    MatrixXd            noi_coefs_f_new_trans(dim, n_dofs_v_per_facet/dim);  // normal interpolada em n+1

    MatrixXd            s_coefs_f_old(n_dofs_s_per_facet, dim);
    MatrixXd            s_coefs_f_new(n_dofs_s_per_facet, dim);
    MatrixXd            s_coefs_f_mid_trans(dim, n_dofs_s_per_facet); // n+utheta

    MatrixXd            vs_coefs_f_new(n_dofs_u_per_facet/dim, dim);        // n+1
    MatrixXd            vs_coefs_f_mid_trans(dim,n_dofs_u_per_facet/dim);   // n+1

    Tensor              F_f_mid(dim,dim-1);       // n+utheta
    Tensor              invF_f_mid(dim-1,dim);    // n+utheta
    Tensor              fff_f_mid(dim-1,dim-1);   // n+utheta; fff = first fundamental form

    MatrixXd            Aloc_f(n_dofs_u_per_facet, n_dofs_u_per_facet);
    MatrixXd            AZaux_f(dim,n_dofs_u_per_facet);
    VectorXd            FUloc_f(n_dofs_u_per_facet), FSloc(LZ), Fval(1);
    VectorXd            FZloc_f(nodes_per_facet*LZ);
    VectorXd            FZsloc_f(LZ);// = VectorXd::Zero(LZ);

    MatrixXd            Z1loc_f(n_dofs_u_per_facet,LZ);
    MatrixXd            Z2loc_f(n_dofs_z_per_facet,n_dofs_u_per_facet);
    MatrixXd            Z3loc_f(n_dofs_z_per_facet,LZ);
    MatrixXd            Z2sloc_f(LZ,n_dofs_u_per_facet);
    MatrixXd            Z3sloc_f(LZ,LZ);

    MatrixXd            Prj(n_dofs_u_per_facet,n_dofs_u_per_facet);
    VectorXi            facet_nodes(nodes_per_facet);

    Vector              normal(dim), traction_(dim), tangent(dim);
    MatrixXd            dxphi_f(n_dofs_u_per_facet/dim, dim);
    Tensor              dxU_f(dim,dim);   // grad u
    Vector              Xqp(dim), Xqpc(3), maxw(3);
    Vector              Uqp(dim), Eqp(dim), Usqp(dim);
    Vector              noi(dim); // normal interpolada
    double              J_mid = 0, JxW_mid, DFtau; //delta_ij,
    double              weight = 0, perE = 0;

    Vector3d            Xg, XIg;
    Vector              XIp(dim), RotfI(dim), RotfJ(dim);
    TensorZ const       IdZ(Tensor::Identity(LZ,LZ));
    MatrixXd            HqT(TensorZ::Zero(LZ,dim));
    VectorXd            Ftau(dim);
    double const*       Xqpb;  //coordonates at the master element \hat{X}
    Vector              Phi(dim), DPhi(dim), X0(dim), X2(dim), Xcc(3), Vdat(3);
    Tensor              F_f_curv(dim,dim-1);
    bool                curvf = false;
    double              ybar = 0.0, areas = 0.0;

    facet_iterator facet = mesh->facetBegin();
    facet_iterator facet_end = mesh->facetEnd();  // the next if controls the for that follows

    if (neumann_tags.size() != 0 || interface_tags.size() != 0 || solid_tags.size() != 0
                                 || slipvel_tags.size() != 0 || flusoli_tags.size() != 0)
    for (; facet != facet_end; ++facet)
    {
      tag = facet->getTag();
      is_neumann = is_in(tag, neumann_tags);
      is_surface = is_in(tag, interface_tags);
      is_solid   = is_in(tag, solid_tags);
      is_slipvel = is_in_id(tag, slipvel_tags);
      is_fsi     = is_in_id(tag, flusoli_tags);

      if ((!is_neumann) && (!is_surface) && (!is_solid) && !(is_slipvel) && !(is_fsi))
        continue;

      // mapeamento do local para o global //////////////////////////////////////////////////
      dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(mapM_f.data(), &*facet);  //cout << mapM_f << endl;
      dof_handler[DH_UNKM].getVariable(VAR_U).getFacetDofs(mapU_f.data(), &*facet);  //cout << mapU_f << endl << endl;  //unk. global ID's
      dof_handler[DH_UNKM].getVariable(VAR_P).getFacetDofs(mapP_f.data(), &*facet);  //cout << mapP_f << endl;

      if (is_fsi || is_slipvel){
        mesh->getFacetNodesId(&*facet, pts_id);
        mapZ_f = -VectorXi::Ones(n_dofs_z_per_facet);
        mapU_t = -VectorXi::Ones(n_dofs_u_per_facet);
        //if (is_unksv){mapUs_f = -(mapU_f+ n_unknowns_ups*VectorXi::Ones(n_dofs_u_per_facet))/*VectorXi::Ones(n_dofs_u_per_cell)*/;}
       //SVI = false; VSF = false;
        for (int j = 0; j < nodes_per_facet; ++j){
          //tag_p = mesh->getNodePtr(pts_id[j])->getTag();  //cout << pts_id[j] << "   ";
          nod_id = is_in_id(tag,flusoli_tags);
          nod_sv = is_in_id(tag,slipvel_tags);
          nodsum = nod_id+nod_sv;
          for (int l = 0; l < LZ; l++){
            mapZ_f(j*LZ + l) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + l;
          }
          for (int l = 0; l < dim; l++){
            mapU_t(j*dim + l) = mapU_f(j*dim + l); //mapU_f(j*dim + l) = -1;
            //if (is_unksv){mapUs_f(j*dim + l) = mapU_f(j*dim + l) + n_unknowns_ups/*-mapUs_c(i*dim + l)*/;}
          }
          //if (nod_sv){SVI = true;}
          //if (nod_id){VSF = true;}
          SV_f[j] = nod_sv; VS_f[j] = nod_id;
        }
      }

      if (is_fsi){
        for (int C = 0; C < LZ; C++){
          mapZ_s(C) = n_unknowns_u + n_unknowns_p + LZ*(is_fsi-1) + C;
        }// works with any SV_f[j]+VS_f[j] because all nodes of the facet are in the same body
      }

      if (is_curvt){
        if (is_fsi || is_slipvel){curvf = true;}
        else {curvf = false;}
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      //get the value for the variables//////////////////////////////////////////////////
      VecGetValues(Vec_normal,  mapM_f.size(), mapM_f.data(), noi_coefs_f_new.data());
      VecGetValues(Vec_x_0,     mapM_f.size(), mapM_f.data(), x_coefs_f_old.data());
      VecGetValues(Vec_x_1,     mapM_f.size(), mapM_f.data(), x_coefs_f_new.data());
      VecGetValues(Vec_uzp_0,   mapU_f.size(), mapU_f.data(), u_coefs_f_old.data());
      VecGetValues(Vec_uzp_k,   mapU_f.size(), mapU_f.data(), u_coefs_f_new.data());
      if (is_fsi || is_slipvel){
        VecGetValues(Vec_uzp_k,   mapZ_f.size(), mapZ_f.data(), z_coefs_f_new.data()); //cout << z_coefs_c_new << endl << endl;
        //if (is_unksv){
        //  VecGetValues(Vec_uzp_k,  mapUs_f.size(), mapUs_f.data(), vs_coefs_f_new.data()); // bdf2,bdf3
        //}
      }

      x_coefs_f_old_trans = x_coefs_f_old.transpose();
      x_coefs_f_new_trans = x_coefs_f_new.transpose();
      u_coefs_f_old_trans = u_coefs_f_old.transpose();
      u_coefs_f_new_trans = u_coefs_f_new.transpose();

      u_coefs_f_mid_trans = utheta*u_coefs_f_new_trans + (1.-utheta)*u_coefs_f_old_trans;
      x_coefs_f_mid_trans = utheta*x_coefs_f_new_trans + (1.-utheta)*x_coefs_f_old_trans;
      if (is_fsi || is_slipvel){
        z_coefs_f_new_trans = z_coefs_f_new.transpose();
        z_coefs_f_mid_trans = z_coefs_f_new_trans;
        //if (is_unksv){
        //  vs_coefs_f_mid_trans = vs_coefs_f_new.transpose();
        //}
      }

      if (curvf){
        Xcc = XG_0[is_fsi+is_slipvel-1];
        X0(0) = x_coefs_f_mid_trans(0,0);  X0(1) = x_coefs_f_mid_trans(1,0);
        X2(0) = x_coefs_f_mid_trans(0,1);  X2(1) = x_coefs_f_mid_trans(1,1);  //cout << Xcc.transpose() << "   " << X0.transpose() << " " << X2.transpose() << endl;
        Vdat << RV[is_fsi+is_slipvel-1](0),RV[is_fsi+is_slipvel-1](1), 0.0; //theta_0[nod_id-1]; //container for R1, R2, theta
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      //initialization as zero of the residuals and elemental matrices////////////////////////////////////////
      FUloc_f.setZero();  //U momentum
      FZloc_f.setZero();  //S momentum
      FZsloc_f.setZero(); //Lagrangian contribution to S momentum

      Aloc_f.setZero();
      Z2sloc_f.setZero(); Z3sloc_f.setZero();
      Z1loc_f.setZero(); Z2loc_f.setZero(); Z3loc_f.setZero();

      if (is_sslv && false){ //for electrophoresis, TODO
        dof_handler[DH_SLIP].getVariable(VAR_S).getFacetDofs(mapS_f.data(), &*facet);
        VecGetValues(Vec_slipv_0, mapS_f.size(), mapS_f.data(), s_coefs_f_old.data());
        VecGetValues(Vec_slipv_1, mapS_f.size(), mapS_f.data(), s_coefs_f_new.data());
        s_coefs_f_mid_trans = utheta*s_coefs_f_new.transpose() + (1.-utheta)*s_coefs_f_old.transpose();;
        FSloc.setZero();
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
      for (int qp = 0; qp < n_qpts_facet; ++qp)
      {

        F_f_mid = Tensor::Zero(dim,dim-1);  //Zero(dim,dim);
        //F_c_old = Tensor::Zero(dim,dim);  //Zero(dim,dim);
        //F_c_new = Tensor::Zero(dim,dim);  //Zero(dim,dim);
        Xqp     = Vector::Zero(dim);// coordenada espacial (x,y,z) do ponto de quadratura
        if (curvf){//F_c_curv.setZero();
          Xqpb = quadr_facet->point(qp);  //cout << Xqpb[0] << " " << Xqpb[1] << endl;
          ybar = (1.0+Xqpb[0])/2.0;
          Phi = curved_Phi(ybar,X0,X2,Xcc,Vdat,dim);
          DPhi = Dcurved_Phi(ybar,X0,X2,Xcc,Vdat,dim);
          //F_f_curv.col(0) = -Phi;
          F_f_curv.col(0) = -Phi/2.0 + (1.0-ybar)*DPhi/2.0;
          F_f_mid = F_f_curv;
          Xqp     = (1.0-ybar)*Phi;
        }

        F_f_mid   += x_coefs_f_mid_trans * dLqsi_f[qp];  // (dim x nodes_per_facet) (nodes_per_facet x dim-1)
        fff_f_mid.resize(dim-1,dim-1);
        fff_f_mid  = F_f_mid.transpose()*F_f_mid;
        J_mid      = sqrt(fff_f_mid.determinant());
        invF_f_mid = fff_f_mid.inverse()*F_f_mid.transpose();

        Xqp    += x_coefs_f_mid_trans * qsi_f[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
        weight  = quadr_facet->weight(qp);
        JxW_mid = J_mid*weight;
        if (is_axis){
          JxW_mid = JxW_mid*2.0*pi*Xqp(0);
        }
        dxphi_f = dLphi_f[qp] * invF_f_mid;
        dxU_f   = u_coefs_f_mid_trans * dxphi_f; // n+utheta
        Uqp     = u_coefs_f_mid_trans * phi_f[qp];
        if (is_fsi && is_unksv){//.col(0) is choosen because the S unknown is the same for any node: they are on the same body
          Usqp  = Uqp - 1.0*SolidVel(Xqp, XG_0[is_fsi-1], z_coefs_f_mid_trans.col(0), dim); //vs_coefs_f_mid_trans * phi_f[qp];
        }
        //noi     = noi_coefs_f_new_trans * qsi_f[qp];
        if (is_sslv && false){ //for electrophoresis, TODO
          Eqp = s_coefs_f_mid_trans * qsi_f[qp];
          perE = per_Elect(tag);
        }

        //calculating normal to the facet//////////////////////////////////////////////////
        if (dim==2)
        {
          if (curvf){
            int K = is_fsi + is_slipvel;
            normal = exact_normal_ellipse(Xqp,XG_0[K-1],0.0,RV[K-1](0),RV[K-1](1),dim); //theta_ini[is_fsiid+is_slvid-1]
            //normal = -normal; //there is no need to do this because normal is alreday OUT the body
            tangent(0) = -normal(1); tangent(1) = normal(0);
          }
          else{//originally this normal points INTO the body (CHECKED!)...
            normal(0) = +F_f_mid(1,0);
            normal(1) = -F_f_mid(0,0);
            normal.normalize();  //cout << normal.transpose() << endl;
            normal = -normal;  //... now this normal points OUT the body
            tangent(0) = -normal(1); tangent(1) = normal(0);
          }
        }
        else
        {
          normal = cross(F_f_mid.col(0), F_f_mid.col(1));
          normal.normalize();
        }
        //////////////////////////////////////////////////

        if (is_neumann) //////////////////////////////////////////////////
        {
          //Vector no(Xqp);
          //no.normalize();
          //traction_ = utheta*(traction(Xqp,current_time+dt,tag)) + (1.-utheta)*traction(Xqp,current_time,tag);
          traction_ = traction(Xqp, normal, current_time + dt*utheta,tag);
          //traction_ = (traction(Xqp,current_time,tag) +4.*traction(Xqp,current_time+dt/2.,tag) + traction(Xqp,current_time+dt,tag))/6.;

          for (int i = 0; i < n_dofs_u_per_facet/dim; ++i)
          {
            for (int c = 0; c < dim; ++c)
            {
              FUloc_f(i*dim + c) -= JxW_mid * traction_(c) * phi_f[qp][i] ; // força
            }
          }
        }//end is_neumann //////////////////////////////////////////////////

        if (is_surface) //////////////////////////////////////////////////
        {
          for (int i = 0; i < n_dofs_u_per_facet/dim; ++i)
          {
            for (int c = 0; c < dim; ++c)
            {
              //FUloc_f(i*dim + c) += JxW_mid *gama(Xqp,current_time,tag)*(dxphi_f(i,c) + (unsteady*dt) *dxU_f.row(c).dot(dxphi_f.row(i))); // correto
              FUloc_f(i*dim + c) += JxW_mid *gama(Xqp,current_time,tag)*dxphi_f(i,c); //inicialmente descomentado
              //FUloc_f(i*dim + c) += JxW_mid *gama(Xqp,current_time,tag)*normal(c)* phi_f[qp][i];
              //for (int d = 0; d < dim; ++d)
              //  FUloc_f(i*dim + c) += JxW_mid * gama(Xqp,current_time,tag)* ( (c==d?1:0) - noi(c)*noi(d) )* dxphi_f(i,d) ;
              //FUloc_f(i*dim + c) += JxW_mid * gama(Xqp,current_time,tag)* ( unsteady*dt *dxU_f.row(c).dot(dxphi_f.row(i)));
            }
          }

          if (true) // semi-implicit term //inicialmente false
          {
            for (int i = 0; i < n_dofs_u_per_facet/dim; ++i)
              for (int j = 0; j < n_dofs_u_per_facet/dim; ++j)
                for (int c = 0; c < dim; ++c)
                  Aloc_f(i*dim + c, j*dim + c) += utheta*JxW_mid* (unsteady*dt) *gama(Xqp,current_time,tag)*dxphi_f.row(i).dot(dxphi_f.row(j));
          }//end semi-implicit
        }//end is_surface //////////////////////////////////////////////////

        if (is_sslv && is_slipvel && false) //for electrophoresis, TODO //////////////////////////////////////////////////
        {
          sid = is_slipvel;

          Xqpc.setZero();
          Xqpc(0) = Xqp(0); Xqpc(1) = Xqp(1); if (dim == 3){Xqpc(2) = Xqp(2);}

          maxw = JxW_mid*traction_maxwell(Eqp, normal, perE, tag);
          FSloc(0) = maxw(0); FSloc(1) = maxw(1); if (dim == 3){FSloc(2) = maxw(2);}

          maxw = JxW_mid*cross((XG_mid[sid]-Xqpc),traction_maxwell(Eqp, normal, perE, tag));
          if (dim == 2){
            FSloc(2) = maxw(2);
          }
          else{
            FSloc.tail(3) = maxw;
          }
        }//end is_sslv //////////////////////////////////////////////////

        if (is_fsi && true) // for Neumann on the body //////////////////////////////////////////////////
        {
          int K = is_fsi;
          Xg = XG_mid[K-1];
          Ftau = force_Ftau(Xqp, Xg, normal, dim, tag, theta_1[K-1], Usqp); //normal points OUT the fluid, see definition of normal above
          DFtau = Dforce_Ftau(Xqp, Xg, normal, dim, tag, theta_1[K-1], Usqp); //cout << Ftau.transpose() << "       " << DFtau << endl;

          //from velocity test functions (see variational formulation)//////////////////////////////////////////////////
          for (int i = 0; i < n_dofs_u_per_facet/dim; ++i)
          {
            for (int c = 0; c < dim; ++c)
            {// force for fluid momentum equation
              FUloc_f(i*dim + c) += JxW_mid * Ftau(c) * phi_f[qp][i]; // force, "+" because Ftau appears "+" in the formulation
            }
          }
          if (is_unksv){
            for (int i = 0; i < n_dofs_u_per_facet/dim; ++i){
              for (int j = 0; j < n_dofs_u_per_facet/dim; ++j){
                for (int c = 0; c < dim; ++c){
                  for (int d = 0; d < dim; ++d){
                    Aloc_f(i*dim + c, j*dim + d) += utheta*JxW_mid*DFtau*phi_f[qp][i]*phi_f[qp][j] * tangent(c)*tangent(d);
                  }
                }
              }
            }
          }

          //for (int c = 0; c < dim; ++c){
          //  RotfJ  = SolidVel(Xqp, XIg, IdZ.col(D), dim);
          //  Hq_vs(i*dim+c,i*LZ) = RotfJ(c);  //builds the elemental matrix H of solid generator movement
          //}

          //from dofs test functions (see variational formulation)//////////////////////////////////////////////////
          HqT       = SolidVelMatrix(Xqp,Xg,dim,LZ).transpose();
          FZsloc_f -= JxW_mid*HqT*Ftau; //cout << HqT*Ftau << endl;
          if (is_unksv){
            for (int j = 0; j < n_dofs_u_per_facet/dim; ++j){
              for (int d = 0; d < dim; ++d){
                for (int c = 0; c < dim; c++){
                  AZaux_f(c,j*dim+d) = phi_f[qp][j]*tangent(c)*tangent(d);
                }
              }
            }
            Z2sloc_f -= JxW_mid*DFtau*HqT*AZaux_f;  //body's dofs equation in lagrangian part
          }

        }// end Neumann on the body //////////////////////////////////////////////////

        if (is_fsi || is_slipvel){areas += JxW_mid;}
      }
      ////////////////////////////////////////////////// ENDING QUADRATURE //////////////////////////////////////////////////

      // Projection - to force non-penetrarion bc //////////////////////////////////////////////////
      if (is_fsi /*SVI || VSF*/){
        VectorXd    FUloc_copy(n_dofs_u_per_facet);
        MatrixXd    Aloc_copy(n_dofs_u_per_facet, n_dofs_u_per_facet);

        FUloc_copy = FUloc_f;
        Aloc_copy  = Aloc_f;

        mesh->getFacetNodesId(&*facet, facet_nodes.data());
        getProjectorMatrix(Prj, nodes_per_facet, facet_nodes.data(), Vec_x_1, current_time+dt, *this);
        FUloc_f = Prj*FUloc_f;  //momentum
        Aloc_f  = Prj*Aloc_f;   //momentum

        MatrixXd    Id_vs(MatrixXd::Identity(n_dofs_u_per_facet,n_dofs_u_per_facet));
        MatrixXd    Hq_vs(MatrixXd::Zero(n_dofs_u_per_facet,n_dofs_z_per_facet));
        MatrixXd    HqS(MatrixXd::Zero(n_dofs_u_per_facet,LZ));
        for (int i = 0; i < n_dofs_u_per_facet/dim; ++i)
        {
          int K = SV_f[i] + VS_f[i];
          XIp    = x_coefs_f_mid_trans.col(i); //ref point Xp, old, mid, or new
          XIg    = XG_mid[K-1];          //mass center, mid, _0, "new"
          //RotfI  = SolidVel(XIp, XIg, z_coefs_f_mid_trans.col(i), dim);

          for (int D = 0; D < LZ; D++){
            RotfI  = SolidVel(XIp, XIg, IdZ.col(D),dim);
            for (int c = 0; c < dim; ++c){
              Hq_vs(i*dim+c,i*LZ+D) = RotfI(c);  //builds the elemental matrix H of solid generator movement
              HqS(i*dim+c,D) = RotfI(c);
            }
          }

          //for (int c = 0; c < dim; ++c)
          //{
          //  for (int D = 0; D < LZ; D++){
          //    RotfI  = SolidVel(XIp, XIg, IdZ.col(D), dim);
          //    Hq_vs(i*dim+c,i*LZ+D) = RotfI(c);  //builds the elemental matrix H of solid generator movement
          //    HqS(i*dim+c,D) = RotfI(c);
          //  }
          //}
        }
        //cout << Id_vs << endl;
        FZloc_f = Hq_vs.transpose()*(Id_vs - Prj)*FUloc_copy; //cout << FZloc_f << endl; //body's dofs equation

        if(is_unksv){
          Z1loc_f  = -Prj*Aloc_copy*HqS;  //minus from the derivative with respect to body's dofs
          Z2loc_f  =  Hq_vs.transpose()*(Id_vs - Prj)*Aloc_copy;
          Z3loc_f  = -Z2loc_f*HqS;//Hq_vs.transpose()*(Id_vs - Prj)*Aloc_copy*HqS;
          Z3sloc_f = -Z2sloc_f*HqS;
        }
      }

      if (true /*is_axis && is_sfip*/){// eliminating specific solid DOFS
        MatrixXd PrjDOFS(n_dofs_z_per_facet,n_dofs_z_per_facet);
        MatrixXd PrjDOFSlz(LZ,LZ);
        MatrixXd Id_LZ(MatrixXd::Identity(n_dofs_z_per_facet,n_dofs_z_per_facet));
        VectorXi s_DOFS(LZ); s_DOFS = DOFS_elimination(LZ); //<< 0, 1, 0;
        getProjectorDOFS(PrjDOFS, n_dofs_u_per_facet/dim, s_DOFS.data(), *this);
        getProjectorDOFS(PrjDOFSlz, 1, s_DOFS.data(), *this);
        FZloc_f = PrjDOFS*FZloc_f;
        FZsloc_f = PrjDOFSlz*FZsloc_f;
        if (is_unksv){
          Z2loc_f = PrjDOFS*Z2loc_f;
          Z3loc_f = PrjDOFS*Z3loc_f;
          Z2sloc_f = PrjDOFSlz*Z2sloc_f;
          Z3sloc_f = PrjDOFSlz*Z3sloc_f;
        }
      }

      //~ FEP_PRAGMA_OMP(critical)
      {
        VecSetValues(Vec_fun_fs, mapU_f.size(), mapU_f.data(), FUloc_f.data(), ADD_VALUES);
        MatSetValues(*JJ, mapU_f.size(), mapU_f.data(), mapU_f.size(), mapU_f.data(), Aloc_f.data(),  ADD_VALUES);
        if (is_fsi && true){
          VecSetValues(Vec_fun_fs, mapZ_f.size(), mapZ_f.data(), FZloc_f.data(), ADD_VALUES);
          VecSetValues(Vec_fun_fs, mapZ_s.size(), mapZ_s.data(), FZsloc_f.data(), ADD_VALUES);
          if (is_unksv){
            MatSetValues(*JJ, mapU_f.size(), mapU_f.data(), mapZ_s.size(), mapZ_s.data(), Z1loc_f.data(), ADD_VALUES);
            MatSetValues(*JJ, mapZ_f.size(), mapZ_f.data(), mapU_f.size(), mapU_f.data(), Z2loc_f.data(), ADD_VALUES);
            MatSetValues(*JJ, mapZ_f.size(), mapZ_f.data(), mapZ_s.size(), mapZ_s.data(), Z3loc_f.data(), ADD_VALUES);
            MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapU_f.size(), mapU_f.data(), Z2sloc_f.data(), ADD_VALUES);
            MatSetValues(*JJ, mapZ_s.size(), mapZ_s.data(), mapZ_s.size(), mapZ_s.data(), Z3sloc_f.data(), ADD_VALUES);
          }
        }
      }
    }// end for facet
    //if (is_axis){cout << "Area = " << areas << ", Area dif = "<< areas-4.0*pi << endl;}
    //else        {cout << "Area = " << areas << ", Area dif = "<< areas-pi << endl;}
  }
  // end LOOP NAS FACES DO CONTORNO (Neum, Interf, Sol) //////////////////////////////////////////////////

  if (force_pressure) //null_space_press_dof calculated at the beginning
  {
    double const p = 1.0;
    MatSetValues(*JJ, 1, &null_space_press_dof, 1, &null_space_press_dof, &p, ADD_VALUES);
  }

  Assembly(Vec_fun_fs);
  Assembly(*JJ);

  if(print_to_matlab)
  {
    static bool ja_foi=false;
    if (!ja_foi)
    {
      char resi[PETSC_MAX_PATH_LEN], jaco[PETSC_MAX_PATH_LEN];
      sprintf(resi,"%s/matrices/rhs.m",filehist_out.c_str());
      sprintf(jaco,"%s/matrices/jacob.m",filehist_out.c_str());
      View(Vec_fun_fs,resi,"Res");
      View(*JJ,jaco,"Jac");
    }
    ja_foi = true;

  }
  //View(Vec_fun_fs, "matrizes/rhs.m","res"); View(*JJ,"matrizes/jacob.m","Jac");
  //MatZeroEntries(*JJ); SNESGetJacobian(snes_fs, JJ, NULL, NULL, NULL); Assembly(*JJ);
  //double val; VecNorm(Vec_fun_fs,NORM_2,&val); cout << "norma residuo " << val <<endl;

  PetscFunctionReturn(0);

} // END formFunction

PetscErrorCode AppCtx::formJacobian_fs(SNES snes_fs,Vec Vec_uzp_k, Mat* /*Mat_Jac*/, Mat* /*prejac*/, MatStructure * /*flag*/)
{
  PetscBool          found = PETSC_FALSE;
  char               snes_type[PETSC_MAX_PATH_LEN];

  PetscOptionsGetString(PETSC_NULL,"-snes_type",snes_type,PETSC_MAX_PATH_LEN-1,&found);

  if (found)
    if (string(snes_type) == string("test"))
    {
      cout << "WARNING: TESTING JACOBIAN !!!!! \n";
      this->formFunction_fs(snes_fs, Vec_uzp_k, Vec_res_fs);
    }

  PetscFunctionReturn(0);
}


// ******************************************************************************
//                            FORM FUNCTION_SQRM
// ******************************************************************************
PetscErrorCode AppCtx::formFunction_sqrm(SNES /*snes_m*/, Vec Vec_v, Vec Vec_fun)
{
  double utheta = AppCtx::utheta;

  if (is_bdf2)
  {
    if (time_step == 0)
      if (!is_bdf_euler_start)
        utheta = 0.5;
  }
  else if (is_bdf3)
  {
    if (time_step <= 1)
      utheta = 0.5;
  }
  //else if (is_basic)
  //  utheta = 0.0;

  utheta = 1.0;
  // NOTE: solve elasticity problem in the mesh at time step n
  // NOTE: The mesh used is the Vec_x_0
  // WARNING: this function assumes that the boundary conditions was already applied

  Mat *JJ = &Mat_Jac_s;
  VecZeroEntries(Vec_fun);
  MatZeroEntries(*JJ);

// LOOP NAS CÉLULAS Parallel (uncomment it)
//#ifdef FEP_HAS_OPENMP
//  FEP_PRAGMA_OMP(parallel default(none) shared(Vec_v, Vec_fun, cout, JJ, utheta))
//#endif
  {
    int               tag;

    Vector            dxS(dim);  // grad u
    Tensor            F_c(dim,dim);
    Tensor            invF_c(dim,dim);
    Tensor            invFT_c(dim,dim);
    double            Sqp;
    Vector            s_coefs_c_trans(n_dofs_s_per_cell);  // mesh velocity;
    Vector            s_coefs_c(n_dofs_s_per_cell);
    MatrixXd          x_coefs_c_trans(dim, nodes_per_cell);
    MatrixXd          x_coefs_c(nodes_per_cell, dim);
    MatrixXd          x_coefs_c_new_trans(dim, nodes_per_cell);
    MatrixXd          x_coefs_c_new(nodes_per_cell, dim);
    MatrixXd          dxpsi_c(n_dofs_s_per_cell, dim);
    double            J, weight, JxW;

    VectorXd          Floc(n_dofs_s_per_cell);
    MatrixXd          Aloc(n_dofs_s_per_cell, n_dofs_s_per_cell);

    VectorXi          mapS_c(n_dofs_s_per_cell); //mapU_c(n_dofs_u_per_cell); // i think is n_dofs_v_per_cell
    VectorXi          mapM_c(dim*nodes_per_cell);

    MatrixXd          Prj(n_dofs_s_per_cell, n_dofs_s_per_cell);
    VectorXi          cell_nodes(nodes_per_cell);
    Point             const* point;
    int               tags, ctags = 0;

    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    cell_iterator cell = mesh->cellBegin(tid,nthreads);   //cell_iterator cell = mesh->cellBegin();
    cell_iterator cell_end = mesh->cellEnd(tid,nthreads); //cell_iterator cell_end = mesh->cellEnd();
    for (; cell != cell_end; ++cell)
    {

      dof_handler[DH_SLIP].getVariable(VAR_S).getCellDofs(mapS_c.data(), &*cell);
      tag = cell->getTag();

      if(is_in(tag,solidonly_tags)){
        ctags = 0;
        for (int i = 0; i < n_dofs_s_per_cell; i++){
          point = mesh->getNodePtr(mapS_c(i));
          tags = point->getTag();
          if (is_in(tags,solidonly_tags)){
            ctags++;
          }
        }
        if (ctags == n_dofs_s_per_cell){
          Aloc.setIdentity();
          #ifdef FEP_HAS_OPENMP
            FEP_PRAGMA_OMP(critical)
          #endif
          {
            MatSetValues(*JJ, mapS_c.size(), mapS_c.data(), mapS_c.size(), mapS_c.data(), Aloc.data(), INSERT_VALUES);
          }
          continue;
        }
      }

      dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);  //cout << mapV_c.transpose() << endl;

      /* Pega os valores das variáveis nos graus de liberdade */
      VecGetValues(Vec_v ,  mapS_c.size(), mapS_c.data(), s_coefs_c.data());  //cout << v_coefs_c << endl;//VecView(Vec_v,PETSC_VIEWER_STDOUT_WORLD);
      VecGetValues(Vec_x_0, mapM_c.size(), mapM_c.data(), x_coefs_c.data());  //cout << x_coefs_c << endl;
      VecGetValues(Vec_x_1, mapM_c.size(), mapM_c.data(), x_coefs_c_new.data());  //cout << x_coefs_c_new << endl;

      if ((is_bdf2 && time_step > 0) || (is_bdf3 && time_step > 1)) //the integration geometry is \bar{X}^{n+1}
        x_coefs_c = x_coefs_c_new;
      else
        x_coefs_c = (1.-utheta)*x_coefs_c + utheta*x_coefs_c_new; // for MR-AB, this completes the geom extrap

      s_coefs_c_trans = s_coefs_c;
      x_coefs_c_trans = x_coefs_c.transpose();

      Floc.setZero();
      Aloc.setZero();

      // Quadrature
      for (int qp = 0; qp < n_qpts_cell; ++qp)
      {
        F_c = x_coefs_c_trans * dLqsi_c[qp];  //cout << dLqsi_c[qp] << endl;
        inverseAndDet(F_c,dim,invF_c,J);
        invFT_c= invF_c.transpose();  //usado?

        dxpsi_c = dLqsi_c[qp] * invF_c;  //dLpsi_c

        dxS = dxpsi_c.transpose() * s_coefs_c_trans; //v_coefs_c_trans * dxqsi_c;       // n+utheta
        Sqp = s_coefs_c_trans.dot(qsi_c[qp]); //v_coefs_c_trans * qsi_c[qp]; //psi_c
        //Xqp      = x_coefs_c_trans * qsi_c[qp]; // coordenada espacial (x,y,z) do ponto de quadratura

        weight = quadr_cell->weight(qp);
        JxW = J*weight;  //parece que no es necesario, ver 2141 (JxW/JxW)

        for (int i = 0; i < n_dofs_s_per_cell; ++i)  //sobre cantidad de funciones de forma
        {
          Floc(i) += -Dif_coeff(tag)*dxS.dot(dxpsi_c.row(i)) * JxW;

          for (int j = 0; j < n_dofs_s_per_cell; ++j)
          {
            Aloc(i,j) += -Dif_coeff(tag)*dxpsi_c.row(j).dot(dxpsi_c.row(i)) * JxW;
            //if (i == j && x_coefs_c(i,0) == 0 && x_coefs_c(i,1) == -0.125) cout << Aloc(i,i) << endl << mapS_c;
          }
        }

      } // fim quadratura

      // Projection - to force non-penetrarion bc
      mesh->getCellNodesId(&*cell, cell_nodes.data());
      getProjectorSQRM(Prj, nodes_per_cell, cell_nodes.data(), *this /*AppCtx*/);
      Floc = Prj*Floc;  //cout << Floc.transpose() << endl;
      Aloc = Prj*Aloc*Prj;  //zeros at dirichlet nodes (lines and columns)
      //for (int i = 0; i < 3; i++)
      //  if (x_coefs_c(i,0) == 0 && x_coefs_c(i,1) == -0.125)
      //    cout << Aloc << endl << endl;

#ifdef FEP_HAS_OPENMP
      FEP_PRAGMA_OMP(critical)
#endif
      {
        VecSetValues(Vec_fun, mapS_c.size(), mapS_c.data(), Floc.data(), ADD_VALUES);
        MatSetValues(*JJ, mapS_c.size(), mapS_c.data(), mapS_c.size(), mapS_c.data(), Aloc.data(), ADD_VALUES);
      }
    } // end cell loop


  } // end parallel
  //Assembly(*JJ); View(*JJ, "matrizes/ElastOpAntes.m", "JJ");
  //MatView(*JJ,PETSC_VIEWER_STDOUT_WORLD);
  {
    int                 tag;
    bool                is_slipvel, is_fsi;

    VectorXi            mapS_f(n_dofs_s_per_facet);
    VectorXi            mapM_f(dim*nodes_per_facet);

    Vector              s_coefs_f_mid(n_dofs_s_per_facet);  // n+utheta
    Vector              s_coefs_f_old(n_dofs_s_per_facet);        // n
    Vector              s_coefs_f_new(n_dofs_s_per_facet);        // n+1

    MatrixXd            x_coefs_f_mid_trans(dim, n_dofs_v_per_facet/dim); // n+utheta
    MatrixXd            x_coefs_f_old(n_dofs_v_per_facet/dim, dim);       // n
    MatrixXd            x_coefs_f_old_trans(dim, n_dofs_v_per_facet/dim); // n
    MatrixXd            x_coefs_f_new(n_dofs_v_per_facet/dim, dim);       // n+1
    MatrixXd            x_coefs_f_new_trans(dim, n_dofs_v_per_facet/dim); // n+1

    Tensor              F_f_mid(dim,dim-1);       // n+utheta
    Tensor              invF_f_mid(dim-1,dim);    // n+utheta
    Tensor              fff_f_mid(dim-1,dim-1);   // n+utheta; fff = first fundamental form

    MatrixXd            Aloc_f(n_dofs_s_per_facet, n_dofs_s_per_facet);
    VectorXd            Floc_f(n_dofs_s_per_facet);

    MatrixXd            Prj(n_dofs_s_per_facet,n_dofs_s_per_facet);
    VectorXi            facet_nodes(nodes_per_facet);

    double              J_mid = 0, JxW_mid, weight = 0;
//#if(false)
    facet_iterator facet = mesh->facetBegin();
    facet_iterator facet_end = mesh->facetEnd();  // the next if controls the for that follows

    if (slipvel_tags.size() != 0)
    for (; facet != facet_end; ++facet)
    {
      tag = facet->getTag();
      is_slipvel = is_in(tag, slipvel_tags);
      is_fsi     = is_in(tag, flusoli_tags);

      if (!(is_slipvel) && !(is_fsi))
        continue;

      dof_handler[DH_SLIP].getVariable(VAR_S).getFacetDofs(mapS_f.data(), &*facet);  //cout << mapM_f << endl;
      dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(mapM_f.data(), &*facet);  //cout << mapM_f << endl;

      VecGetValues(Vec_x_0,     mapM_f.size(), mapM_f.data(), x_coefs_f_old.data());
      VecGetValues(Vec_x_1,     mapM_f.size(), mapM_f.data(), x_coefs_f_new.data());
      VecGetValues(Vec_v,       mapS_f.size(), mapS_f.data(), s_coefs_f_new.data());

      x_coefs_f_old_trans = x_coefs_f_old.transpose();
      x_coefs_f_new_trans = x_coefs_f_new.transpose();

      s_coefs_f_mid = s_coefs_f_new;
      x_coefs_f_mid_trans = utheta*x_coefs_f_new_trans + (1.-utheta)*x_coefs_f_old_trans;
      //cout << x_coefs_f_mid_trans.transpose() << endl;
      Floc_f.setZero();

      // Quadrature
      for (int qp = 0; qp < n_qpts_facet; ++qp)
      {

        F_f_mid   = x_coefs_f_mid_trans * dLqsi_f[qp];  // (dim x nodes_per_facet) (nodes_per_facet x dim-1)

        fff_f_mid.resize(dim-1,dim-1);
        fff_f_mid  = F_f_mid.transpose()*F_f_mid;
        J_mid      = sqrt(fff_f_mid.determinant());
        invF_f_mid = fff_f_mid.inverse()*F_f_mid.transpose();

        weight  = quadr_facet->weight(qp);
        JxW_mid = J_mid*weight;
        //cout << psi_f[qp].transpose() << "   " << JxW_mid << endl;
        for (int i = 0; i < n_dofs_s_per_facet; ++i)
          Floc_f(i) += nuB_coeff(tag)*sig_coeff(tag)*qsi_f[qp][i] * JxW_mid;  //psi_f


      }//end for Quadrature

      // Projection - to force non-penetrarion bc
      //mesh->getFacetNodesId(&*facet, facet_nodes.data());
      //getProjectorSQRM(Prj, nodes_per_facet, facet_nodes.data(), *this);
      //cout << Floc_f.transpose() << endl;
      //Floc_f = Prj*Floc_f;

      //~ FEP_PRAGMA_OMP(critical)
      {
        VecSetValues(Vec_fun, mapS_f.size(), mapS_f.data(), Floc_f.data(), ADD_VALUES);
      }

    }//end for facet
//#endif
  }

//#if(false)
  // boundary conditions on global Jacobian
    // solid & triple tags .. force normal
  if (force_dirichlet)  //identify the contribution of points in *_tags
  {
    //int      nodeid;
    int      v_dofs[1];
    Vector   normal(dim);
    double   A;//Tensor   A(1,1);
    //Tensor   I(Tensor::Identity(1,1));
    int      tag;

    point_iterator point = mesh->pointBegin();
    point_iterator point_end = mesh->pointEnd();
    for ( ; point != point_end; ++point)
    {
      tag = point->getTag();
      if (!(is_in(tag,feature_tags)   ||
            is_in(tag,solid_tags)     ||
            is_in(tag,interface_tags) ||
            is_in(tag,triple_tags)    ||
            is_in(tag,dirichlet_tags) ||
            is_in(tag,neumann_tags)   ||
            is_in(tag,periodic_tags)  ||
            is_in(tag,flusoli_tags)   ||
            is_in(tag,solidonly_tags) ))
        continue;

      getNodeDofs(&*point, DH_SLIP, VAR_S, v_dofs);
      A = 1.0;
      if ( is_in(tag,dirichlet_tags) || is_in(tag,solidonly_tags) )
        A = 0.0;

      A = 1.0 - A;
      MatSetValues(*JJ, 1, v_dofs, 1, v_dofs, &A, ADD_VALUES);//);INSERT_VALUES
    }
  }
//#endif

  Assembly(*JJ);  //View(*JJ, "matrizes/jac.m", "J"); //MatView(*JJ,PETSC_VIEWER_STDOUT_WORLD);
  Assembly(Vec_fun);  //View(Vec_fun, "matrizes/rhs.m", "R");
  //View(*JJ, "ElastOp", "JJ");
  //double val; VecNorm(Vec_fun,NORM_2,&val); cout << "norma residuo " << val <<endl;
  //cout << "Mesh calculation:" << endl;
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::formJacobian_sqrm(SNES /*snes*/,Vec /*Vec_up_k*/,Mat* /**Mat_Jac*/, Mat* /*prejac*/, MatStructure * /*flag*/)
{
  // jacobian matrix is done in the formFunction_mesh
  PetscFunctionReturn(0);
}


// ******************************************************************************
//                            FORM FUNCTION_FD
// ******************************************************************************
PetscErrorCode AppCtx::formFunction_fd(SNES /*snes_m*/, Vec Vec_fd, Vec Vec_fun)
{
  double utheta = AppCtx::utheta;
  if (is_mr){utheta = 1.0;}

  VecSetOption(Vec_fun, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  VecSetOption(Vec_fd, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);

  Mat *JJ = &Mat_Jac_fd;
  VecZeroEntries(Vec_fun);
  MatZeroEntries(*JJ);

  // LOOP NAS CÉLULAS Parallel (uncomment it) //////////////////////////////////////////////////
#ifdef FEP_HAS_OPENMP
  FEP_PRAGMA_OMP(parallel default(none) shared(Vec_fd,Vec_fun,cout,JJ,utheta))
#endif
  {
    VectorXd            FUloc(n_dofs_u_per_cell);  // U subvector part of F
    VectorXd            FPloc(n_dofs_p_per_cell);
    //VectorXd            FZloc(n_dofs_z_per_cell);

    /* local data */
    int                 tag, tag_c, nod_is, nodsum;
    MatrixXd            u_coefs_c_mid_trans(dim, n_dofs_u_per_cell/dim);  // n+utheta  // trans = transpost
    MatrixXd            u_coefs_c_old(n_dofs_u_per_cell/dim, dim);        // n
    MatrixXd            u_coefs_c_old_trans(dim,n_dofs_u_per_cell/dim);   // n
    MatrixXd            u_coefs_c_new(n_dofs_u_per_cell/dim, dim);        // n+1
    MatrixXd            u_coefs_c_new_trans(dim,n_dofs_u_per_cell/dim);   // n+1

    MatrixXd            v_coefs_c_mid(nodes_per_cell, dim);        // mesh velocity; n
    MatrixXd            v_coefs_c_mid_trans(dim,nodes_per_cell);   // mesh velocity; n

    VectorXd            p_coefs_c_new(n_dofs_p_per_cell);  // n+1
    VectorXd            p_coefs_c_old(n_dofs_p_per_cell);  // n
    VectorXd            p_coefs_c_mid(n_dofs_p_per_cell);  // n

    MatrixXd            x_coefs_c_mid_trans(dim, nodes_per_cell); // n+utheta
    MatrixXd            x_coefs_c_new(nodes_per_cell, dim);       // n+1
    MatrixXd            x_coefs_c_new_trans(dim, nodes_per_cell); // n+1
    MatrixXd            x_coefs_c_old(nodes_per_cell, dim);       // n
    MatrixXd            x_coefs_c_old_trans(dim, nodes_per_cell); // n

    MatrixXd            f_coefs_c_mid_trans(dim, nodes_per_cell); // n+utheta
    MatrixXd            f_coefs_c_new(nodes_per_cell, dim);       // n+1

    Tensor              F_c_mid(dim,dim);       // n+utheta
    Tensor              invF_c_mid(dim,dim);    // n+utheta
    //Tensor              invFT_c_mid(dim,dim);   // n+utheta

    Tensor              F_c_old(dim,dim);       // n
    Tensor              invF_c_old(dim,dim);    // n
    //Tensor              invFT_c_old(dim,dim);   // n

    Tensor              F_c_new(dim,dim);       // n+1
    Tensor              invF_c_new(dim,dim);    // n+1
    //Tensor              invFT_c_new(dim,dim);   // n+1

    VectorXi            mapU_c(n_dofs_u_per_cell);
    VectorXi            mapP_c(n_dofs_p_per_cell);

    // mesh velocity
    VectorXi            mapM_c(dim*nodes_per_cell);
    VectorXi            mapM_t(dim*nodes_per_cell);

    /* All variables are in (n+utheta) by default */
    MatrixXd            dxphi_c(n_dofs_u_per_cell/dim, dim);
    MatrixXd            dxphi_c_new(dxphi_c);
    MatrixXd            dxpsi_c(n_dofs_p_per_cell, dim);
    MatrixXd            dxpsi_c_new(dxpsi_c);
    MatrixXd            dxqsi_c(nodes_per_cell, dim);
    Vector              dxbble(dim);
    Vector              dxbble_new(dim);
    Tensor              dxU(dim,dim), dxZ(dim,dim);   // grad u
    Tensor              dxU_old(dim,dim);   // grad u
    Tensor              dxU_new(dim,dim);   // grad u
    Tensor              dxUb(dim,dim);  // grad u bble
    Vector              dxP_new(dim);   // grad p
    Vector              Xqp(dim);
    Vector              Xqp_old(dim);
    Vector              Xc(dim);  // cell center; to compute CR element
    Vector              Uqp(dim), Zqp(dim);
    Vector              Ubqp(dim); // bble
    Vector              Uqp_old(dim), Zqp_old(dim); // n
    Vector              Uqp_new(dim), Zqp_new(dim); // n+1
    Vector              dUqp_old(dim), Uqp_m1(dim);  // n
    Vector              dUqp_vold(dim), Uqp_m2(dim);  // n
    Vector              Vqp(dim);
    Vector              Uconv_qp(dim);
    Vector              dUdt(dim);
    double              Pqp_new;
    VectorXi            cell_nodes(nodes_per_cell);
    double              J_mid;
    double              J_new, J_old;
    double              JxW_mid;  //JxW_new, JxW_old;
    double              weight;
    double              visc=-1; // viscosity
    double              rho;
    //double              delta_cd;
    Vector              force_at_mid(dim);
    Vector              Fdqp(dim);
    MatrixXd            Prj(dim*nodes_per_cell,dim*nodes_per_cell); // projector matrix

    MatrixXd            Aloc(dim*nodes_per_cell,dim*nodes_per_cell);

    ////////////////////////////////////////////////// STARTING CELL ITERATION //////////////////////////////////////////////////
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    cell_iterator cell = mesh->cellBegin(tid,nthreads);    //cell_iterator cell = mesh->cellBegin();
    cell_iterator cell_end = mesh->cellEnd(tid,nthreads);  //cell_iterator cell_end = mesh->cellEnd();

    for (; cell != cell_end; ++cell)
    {

      tag = cell->getTag();
/*
      // finding pure fluid cell ////////////////////
      int fonly_count = 0;
      for (int i = 0; i < nodes_per_cell; ++i)
      {
        tag_c  = mesh->getNodePtr(cell->getNodeId(i))->getTag();
        if (is_in(tag_c,fluidonly_tags) || is_in(tag_c,dirichlet_tags) || is_in(tag_c,neumann_tags))
          fonly_count++;
      }
      bool pure_fluid_c = false;
      if (fonly_count == nodes_per_cell)
        pure_fluid_c = true;
      ////////////////////
*/
      if(is_in(tag,solidonly_tags) /*|| pure_fluid_c*/){//considers the cells completely inside the solid to impose the Dirichlet Fluid-Solid condition
        FUloc.setZero();
        Aloc.setIdentity();
        mapM_t = -VectorXi::Ones(dim*nodes_per_cell);

        dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);

        for (int i = 0; i < nodes_per_cell; ++i)
        {
          tag_c  = mesh->getNodePtr(cell->getNodeId(i))->getTag();
          nod_is = is_in_id(tag_c,solidonly_tags);
          //nod_sv = is_in_id(tag_c,slipvel_tags);
          nodsum = nod_is; //+nod_sv;
          if (nodsum){
            for (int l = 0; l < dim; l++){
              mapM_t(i*dim + l) = mapM_c(i*dim + l);
            }
          }
        }

        #ifdef FEP_HAS_OPENMP
              FEP_PRAGMA_OMP(critical)
        #endif
        {
          VecSetValues(Vec_fun, mapM_t.size(), mapM_t.data(), FUloc.data(), ADD_VALUES);
          MatSetValues(*JJ, mapM_t.size(), mapM_t.data(), mapM_t.size(), mapM_t.data(), Aloc.data(), ADD_VALUES);
          /*if (pure_fluid_c){
            VecSetValues(Vec_fun, mapM_c.size(), mapM_c.data(), FUloc.data(), ADD_VALUES);
            MatSetValues(*JJ, mapM_c.size(), mapM_c.data(), mapM_c.size(), mapM_c.data(), Aloc.data(), ADD_VALUES);
          }*/
        }
        continue;
      }

      // mapeamento do local para o global //////////////////////////////////////////////////
      dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), &*cell);  //cout << mapM_c.transpose() << endl;  //unk. global ID's
      dof_handler[DH_UNKM].getVariable(VAR_U).getCellDofs(mapU_c.data(), &*cell);  //cout << mapU_c.transpose() << endl;
      dof_handler[DH_UNKM].getVariable(VAR_P).getCellDofs(mapP_c.data(), &*cell);  //cout << mapP_c.transpose() << endl;

      //initialize zero values for the variables//////////////////////////////////////////////////
      u_coefs_c_old = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);
      u_coefs_c_new = MatrixXd::Zero(n_dofs_u_per_cell/dim,dim);

      //get the value for the variables//////////////////////////////////////////////////
      if ((is_bdf2 && time_step > 0) || (is_bdf3 && time_step > 1))
        VecGetValues(Vec_v_1, mapM_c.size(), mapM_c.data(), v_coefs_c_mid.data());
      else
        VecGetValues(Vec_v_mid, mapM_c.size(), mapM_c.data(), v_coefs_c_mid.data());  //cout << v_coefs_c_mid << endl << endl;//size of vector mapM_c

      VecGetValues(Vec_x_0,     mapM_c.size(), mapM_c.data(), x_coefs_c_old.data());  //cout << x_coefs_c_old << endl << endl;
      VecGetValues(Vec_x_1,     mapM_c.size(), mapM_c.data(), x_coefs_c_new.data());  //cout << x_coefs_c_new << endl << endl;
      VecGetValues(Vec_uzp_0,   mapU_c.size(), mapU_c.data(), u_coefs_c_old.data());  //cout << u_coefs_c_old << endl << endl;
      VecGetValues(Vec_uzp_1,   mapU_c.size(), mapU_c.data(), u_coefs_c_new.data());  //cout << u_coefs_c_new << endl << endl;
      VecGetValues(Vec_uzp_0,   mapP_c.size(), mapP_c.data(), p_coefs_c_old.data());  //cout << p_coefs_c_old << endl << endl;
      VecGetValues(Vec_uzp_1,   mapP_c.size(), mapP_c.data(), p_coefs_c_new.data());  //cout << p_coefs_c_new << endl << endl;
      VecGetValues(Vec_fd,      mapM_c.size(), mapM_c.data(), f_coefs_c_new.data());  //cout << x_coefs_c_new << endl << endl;

      v_coefs_c_mid_trans = v_coefs_c_mid.transpose();  //cout << v_coefs_c_mid_trans << endl << endl;
      x_coefs_c_old_trans = x_coefs_c_old.transpose();
      x_coefs_c_new_trans = x_coefs_c_new.transpose();
      u_coefs_c_old_trans = u_coefs_c_old.transpose();  //cout << u_coefs_c_old_trans << endl << endl;
      u_coefs_c_new_trans = u_coefs_c_new.transpose();

      u_coefs_c_mid_trans = utheta*u_coefs_c_new_trans + (1.-utheta)*u_coefs_c_old_trans;
      x_coefs_c_mid_trans = utheta*x_coefs_c_new_trans + (1.-utheta)*x_coefs_c_old_trans;
      p_coefs_c_mid       = utheta*p_coefs_c_new       + (1.-utheta)*p_coefs_c_old;
      f_coefs_c_mid_trans = f_coefs_c_new.transpose();

      //viscosity and density at the cell//////////////////////////////////////////////////
      visc = muu(tag);
      rho  = pho(Xqp,tag);  //pho is elementwise, so Xqp does nothing
      //////////////////////////////////////////////////

      //initialization as zero of the residuals//////////////////////////////////////////////////
      FUloc.setZero();
      FPloc.setZero();
      //FZloc.setZero();
      //initialization as zero of the elemental matrices//////////////////////////////////////////////////
      Aloc.setZero();
      //Gloc.setZero();
      //Dloc.setZero();
      //Eloc.setZero();

      ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
      for (int qp = 0; qp < n_qpts_cell; ++qp)
      {

        F_c_mid = x_coefs_c_mid_trans * dLqsi_c[qp];  // (dim x nodes_per_cell) (nodes_per_cell x dim)
        F_c_old = x_coefs_c_old_trans * dLqsi_c[qp];
        F_c_new = x_coefs_c_new_trans * dLqsi_c[qp];
        inverseAndDet(F_c_mid,dim,invF_c_mid,J_mid);
        inverseAndDet(F_c_old,dim,invF_c_old,J_old);
        inverseAndDet(F_c_new,dim,invF_c_new,J_new);
        //invFT_c_mid= invF_c_mid.transpose();
        //invFT_c_old= invF_c_old.transpose();
        //invFT_c_new= invF_c_new.transpose();

        dxphi_c_new = dLphi_c[qp] * invF_c_new;
        dxphi_c     = dLphi_c[qp] * invF_c_mid;
        dxpsi_c_new = dLpsi_c[qp] * invF_c_new;
        dxpsi_c     = dLpsi_c[qp] * invF_c_mid;
        dxqsi_c     = dLqsi_c[qp] * invF_c_mid;

        dxU      = u_coefs_c_mid_trans * dLphi_c[qp] * invF_c_mid; // n+utheta
        dxU_new  = u_coefs_c_new_trans * dLphi_c[qp] * invF_c_new; // n+1
        dxU_old  = u_coefs_c_old_trans * dLphi_c[qp] * invF_c_old; // n
        dxP_new  = dxpsi_c.transpose() * p_coefs_c_new;

        Xqp      = x_coefs_c_mid_trans * qsi_c[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
        Xqp_old  = x_coefs_c_old_trans * qsi_c[qp]; // coordenada espacial (x,y,z) do ponto de quadratura
        Uqp      = u_coefs_c_mid_trans * phi_c[qp]; //n+utheta
        Uqp_new  = u_coefs_c_new_trans * phi_c[qp]; //n+1
        Uqp_old  = u_coefs_c_old_trans * phi_c[qp]; //n
        Pqp_new  = p_coefs_c_new.dot(psi_c[qp]);
//        Pqp      = p_coefs_c_mid.dot(psi_c[qp]);
        Vqp      = v_coefs_c_mid_trans * qsi_c[qp];
        dUdt     = (Uqp_new-Uqp_old)/dt;  //D1f^{n+1}/dt = U^{n+1}/dt-U^{n}/dt
        Uconv_qp = Uqp - Vqp;  //Uconv_qp = Uqp_old;
        Fdqp     = f_coefs_c_mid_trans * qsi_c[qp];

        force_at_mid = force(Xqp,current_time+utheta*dt,tag);

        weight = quadr_cell->weight(qp);
        JxW_mid = J_mid*weight;
        if (is_axis){
          JxW_mid = JxW_mid*2*pi*Xqp(0);
        }

        if (J_mid < 1.e-20)
        {
          FEP_PRAGMA_OMP(critical)
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
            throw;
          }
        }

        // ---------------- //////////////////////////////////////////////////
        //
        //  RESIDUAL AND JACOBIAN MATRIX
        //
        //  --------------- //////////////////////////////////////////////////

        for (int i = 0; i < n_dofs_u_per_cell/dim; ++i)
        {
          for (int c = 0; c < dim; ++c)
          {
            FUloc(i*dim + c) += JxW_mid*  //momentum residual
                 ( //Fdqp(c)*phi_c[qp][i]
                  +rho*(unsteady*dUdt(c) + has_convec*Uconv_qp.dot(dxU.row(c)))*qsi_c[qp][i]  //aceleração
                  +visc*dxqsi_c.row(i).dot(dxU.row(c) + dxU.col(c).transpose())  //rigidez  //transpose() here is to add 2 row-vectors
                  -force_at_mid(c)*qsi_c[qp][i]  //força
                  -Pqp_new*dxqsi_c(i,c) );        //pressão
            if (is_axis && c == 0){
              FUloc(i*dim + c) += JxW_mid*(  //Momentum residual
                    +2.0*visc*qsi_c[qp][i]*Uqp(c)/(Xqp(c)*Xqp(c))  //rigidez
                    -Pqp_new*qsi_c[qp][i]/(Xqp(c)));        //pressão
            }
//            for (int j = 0; j < n_dofs_u_per_cell/dim; ++j)
//            {
//              for (int d = 0; d < dim; ++d)
//              {
//                delta_cd = c==d;
//                Aloc(i*dim + c, j*dim + d) += JxW_mid * delta_cd * phi_c[qp][i]*phi_c[qp][j];
//              }
//            }
          }
        }

      }
      ////////////////////////////////////////////////// ENDING QUADRATURE //////////////////////////////////////////////////

      // Projection - to force Dirichlet conditions: solid motion, non-penetration, pure Dirichlet, respect //////////////////////////////////////////////////
      mesh->getCellNodesId(&*cell, cell_nodes.data());  //cout << cell_nodes.transpose() << endl;
      getProjectorFD(Prj, nodes_per_cell, cell_nodes.data(), Vec_x_1, current_time+dt, *this);
      MatrixXd Id(MatrixXd::Identity(dim*nodes_per_cell,dim*nodes_per_cell));
      FUloc = Prj*FUloc;
      Aloc  = /*Prj*Aloc + */ Id-Prj;

      // ---------------- //////////////////////////////////////////////////
      //
      //  MATRIX ASSEMBLY
      //
      //  --------------- //////////////////////////////////////////////////
#ifdef FEP_HAS_OPENMP
      FEP_PRAGMA_OMP(critical)
#endif
      {
        VecSetValues(Vec_fun, mapM_c.size(), mapM_c.data(), FUloc.data(), ADD_VALUES);
        MatSetValues(*JJ, mapM_c.size(), mapM_c.data(), mapM_c.size(), mapM_c.data(), Aloc.data(), ADD_VALUES);
      }
    }//end for cell

  }//end FEP_HAS_OPENMP

#if (true)
  // LOOP NAS FACES DO CONTORNO (Neum, Interf, Sol, NeumBody) //////////////////////////////////////////////////
  //~ FEP_PRAGMA_OMP(parallel default(none) shared(Vec_uzp_k,Vec_fun_fs,cout))
  {
    int                 tag, is_slipvel, is_fsi;
    bool                is_neumann, is_surface, is_solid;

    MatrixXd            Aloc_f(n_dofs_v_per_facet, n_dofs_v_per_facet);
    VectorXd            FUloc_f(n_dofs_v_per_facet);
    Vector              normal(dim), traction_(dim);
    double              J_mid = 0, JxW_mid;
    Vector              Xqp(dim);
    double              weight = 0, areas = 0;

    VectorXi            mapM_f(dim*nodes_per_facet);

    MatrixXd            x_coefs_f_mid_trans(dim, n_dofs_v_per_facet/dim); // n+utheta
    MatrixXd            x_coefs_f_old(n_dofs_v_per_facet/dim, dim);       // n
    MatrixXd            x_coefs_f_old_trans(dim, n_dofs_v_per_facet/dim); // n
    MatrixXd            x_coefs_f_new(n_dofs_v_per_facet/dim, dim);       // n+1
    MatrixXd            x_coefs_f_new_trans(dim, n_dofs_v_per_facet/dim); // n+1

    MatrixXd            f_coefs_f_mid_trans(dim, n_dofs_v_per_facet/dim); // n+utheta
    MatrixXd            f_coefs_f_old(n_dofs_v_per_facet/dim, dim);       // n
    MatrixXd            f_coefs_f_old_trans(dim, n_dofs_v_per_facet/dim); // n
    MatrixXd            f_coefs_f_new(n_dofs_v_per_facet/dim, dim);       // n+1
    MatrixXd            f_coefs_f_new_trans(dim, n_dofs_v_per_facet/dim); // n+1

    //MatrixXd            noi_coefs_f_new(n_dofs_v_per_facet/dim, dim);  // normal interpolada em n+1
    //MatrixXd            noi_coefs_f_new_trans(dim, n_dofs_v_per_facet/dim);  // normal interpolada em n+1

    Tensor              F_f_mid(dim,dim-1);       // n+utheta
    Tensor              invF_f_mid(dim-1,dim);    // n+utheta
    Tensor              fff_f_mid(dim-1,dim-1);   // n+utheta; fff = first fundamental form
    Vector              Fdqp(dim);

    facet_iterator facet = mesh->facetBegin();
    facet_iterator facet_end = mesh->facetEnd();  // the next if controls the for that follows

    if (neumann_tags.size() != 0 || interface_tags.size() != 0 || solid_tags.size() != 0
                                 || slipvel_tags.size() != 0 || flusoli_tags.size() != 0)
    for (; facet != facet_end; ++facet)
    {
      tag = facet->getTag();
      is_neumann = is_in(tag, neumann_tags);
      is_surface = is_in(tag, interface_tags);
      is_solid   = is_in(tag, solid_tags);
      is_slipvel = is_in_id(tag, slipvel_tags);
      is_fsi     = is_in_id(tag, flusoli_tags);

      if ((!is_neumann) && (!is_surface) && (!is_solid) && !(is_slipvel) && !(is_fsi))
        continue;

      dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(mapM_f.data(), &*facet);  //cout << mapM_f << endl;

      //VecGetValues(Vec_normal,  mapM_f.size(), mapM_f.data(), noi_coefs_f_new.data());
      VecGetValues(Vec_x_0,     mapM_f.size(), mapM_f.data(), x_coefs_f_old.data());
      VecGetValues(Vec_x_1,     mapM_f.size(), mapM_f.data(), x_coefs_f_new.data());
      VecGetValues(Vec_fd,      mapM_f.size(), mapM_f.data(), f_coefs_f_new.data());  //cout << x_coefs_c_new << endl << endl;
//      VecGetValues(Vec_uzp_0,   mapU_f.size(), mapU_f.data(), u_coefs_f_old.data());
//      VecGetValues(Vec_uzp_k,   mapU_f.size(), mapU_f.data(), u_coefs_f_new.data());

      x_coefs_f_old_trans = x_coefs_f_old.transpose();
      x_coefs_f_new_trans = x_coefs_f_new.transpose();
//      u_coefs_f_old_trans = u_coefs_f_old.transpose();
//      u_coefs_f_new_trans = u_coefs_f_new.transpose();

//      u_coefs_f_mid_trans = utheta*u_coefs_f_new_trans + (1.-utheta)*u_coefs_f_old_trans;
      x_coefs_f_mid_trans = utheta*x_coefs_f_new_trans + (1.-utheta)*x_coefs_f_old_trans;
      f_coefs_f_mid_trans = f_coefs_f_new.transpose();

      FUloc_f.setZero();  //U momentum
      Aloc_f.setZero();

      ////////////////////////////////////////////////// STARTING QUADRATURE //////////////////////////////////////////////////
      for (int qp = 0; qp < n_qpts_facet; ++qp)
      {

        F_f_mid   = x_coefs_f_mid_trans * dLqsi_f[qp];  // (dim x nodes_per_facet) (nodes_per_facet x dim-1)
        fff_f_mid.resize(dim-1,dim-1);
        fff_f_mid  = F_f_mid.transpose()*F_f_mid;
        J_mid      = sqrt(fff_f_mid.determinant());
        invF_f_mid = fff_f_mid.inverse()*F_f_mid.transpose();

        Xqp  = x_coefs_f_mid_trans * qsi_f[qp];
        weight  = quadr_facet->weight(qp);
        JxW_mid = J_mid*weight;
        if (is_axis){
          JxW_mid = JxW_mid*2.0*pi*Xqp(0);
        }

        Fdqp      = f_coefs_f_mid_trans * qsi_f[qp];

        //calculating normal to the facet//////////////////////////////////////////////////
        if (dim==2)
        {//originally this normal points INTO the body (CHECKED!)...
          normal(0) = +F_f_mid(1,0);
          normal(1) = -F_f_mid(0,0);
          normal.normalize();  //cout << normal.transpose() << endl;
          normal = -normal;  //... now this normal points OUT the body
          //tangent(0) = -normal(1); tangent(1) = normal(0);
        }
        else
        {
          normal = cross(F_f_mid.col(0), F_f_mid.col(1));
          normal.normalize();
        }
        //////////////////////////////////////////////////

        if (false && is_neumann) //////////////////////////////////////////////////
        {
          //Vector no(Xqp);
          //no.normalize();
          //traction_ = utheta*(traction(Xqp,current_time+dt,tag)) + (1.-utheta)*traction(Xqp,current_time,tag);
          traction_ = traction(Xqp, normal, current_time + dt*utheta,tag);
          //traction_ = (traction(Xqp,current_time,tag) +4.*traction(Xqp,current_time+dt/2.,tag) + traction(Xqp,current_time+dt,tag))/6.;

          for (int i = 0; i < n_dofs_u_per_facet/dim; ++i)
          {
            for (int c = 0; c < dim; ++c)
            {
              FUloc_f(i*dim + c) -= JxW_mid * traction_(c) * qsi_f[qp][i] ; // força
            }
          }
        }//end is_neumann //////////////////////////////////////////////////

        if (is_fsi || is_slipvel){
          for (int i = 0; i < n_dofs_u_per_facet/dim; ++i){
            for (int c = 0; c < dim; ++c){
              FUloc_f(i*dim + c) += JxW_mid * Fdqp(c) * qsi_f[qp][i];
            }
            for (int j = 0; j < n_dofs_u_per_facet/dim; ++j){
              for (int c = 0; c < dim; ++c){
                Aloc_f(i*dim + c, j*dim + c) += JxW_mid * qsi_f[qp][i]*qsi_f[qp][j];
              }
            }
          }

          areas += JxW_mid;
        }//end is_fsi || is_slipvel

      }
      ////////////////////////////////////////////////// ENDING QUADRATURE //////////////////////////////////////////////////

      //~ FEP_PRAGMA_OMP(critical)
      {
        VecSetValues(Vec_fun, mapM_f.size(), mapM_f.data(), FUloc_f.data(), ADD_VALUES);
        MatSetValues(*JJ, mapM_f.size(), mapM_f.data(), mapM_f.size(), mapM_f.data(), Aloc_f.data(),  ADD_VALUES);
      }

    }//end for facet
    //cout << "Areas = " << areas << endl;
  }
  // end LOOP NAS FACES DO CONTORNO (Neum, Interf, Sol) //////////////////////////////////////////////////
#endif

  Assembly(Vec_fun);
  Assembly(*JJ);

  if(print_to_matlab)
  {
    static bool ja_foi=false;
    if (!ja_foi)
    {
      char resi[PETSC_MAX_PATH_LEN], jaco[PETSC_MAX_PATH_LEN];
      sprintf(resi,"%s/matrices/rhs_fd.m",filehist_out.c_str());
      sprintf(jaco,"%s/matrices/jacob_fd.m",filehist_out.c_str());
      View(Vec_fun,resi,"ResFd"); //View(Vec_fun, "matrizes/forcd/rhs.m","Res");
      View(*JJ,jaco,"JacFd"); //View(*JJ,"matrizes/forcd/jacob.m","Jac");
    }
    ja_foi = true;

  }

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::formJacobian_fd(SNES /*snes*/,Vec /*Vec_up_k*/,Mat* /**Mat_Jac*/, Mat* /*prejac*/, MatStructure * /*flag*/)
{
  // jacobian matrix is done in the formFunction_mesh
  PetscFunctionReturn(0);
}
