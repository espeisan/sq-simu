#include "common.hpp"
#include <tr1/tuple>

using std::tr1::tuple;
using std::tr1::make_tuple;
using std::tr1::get;

//extern PetscErrorCode FormJacobian(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
//extern PetscErrorCode FormFunction(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);
extern PetscErrorCode CheckSnesConvergence(SNES snes, PetscInt it,PetscReal xnorm, PetscReal pnorm, PetscReal fnorm, SNESConvergedReason *reason, void *ctx);
extern PetscErrorCode FormJacobian_mesh(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
extern PetscErrorCode FormFunction_mesh(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);
extern PetscErrorCode FormJacobian_fs(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
extern PetscErrorCode FormFunction_fs(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);
extern PetscErrorCode FormJacobian_sqrm(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
extern PetscErrorCode FormFunction_sqrm(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);
//extern PetscErrorCode FormJacobian_fd(SNES snes,Vec Vec_up_1,Mat *Mat_Jac, Mat *prejac, MatStructure *flag, void *ptr);
//extern PetscErrorCode FormFunction_fd(SNES snes, Vec Vec_up_1, Vec Vec_fun, void *ptr);

void checkConsistencyTri(Mesh *mesh)
{
  cell_iterator cell = mesh->cellBegin();
  cell_iterator cell_end = mesh->cellEnd();
  
  int const nfpc = mesh->numFacetsPerCell();
  //  int const nnpc = mesh->numNodesPerCell();
  int const nnpf = mesh->numNodesPerFacet();
  
  int f_nds[nnpf];
  
  //  Point *p;
  Facet *f;
  Cell  *c;
  
  for (; cell != cell_end; ++cell)
  {
    int myid = mesh->getCellId(&*cell);
    for (int i = 0; i < nfpc; ++i)
    {
      if (cell->getIncidCell(i) >= 0)
      {
        // verifica o vizinho
        c = mesh->getCellPtr(cell->getIncidCell(i));
        int pos = cell->getIncidCellPos(i);
        if(myid != c->getIncidCell(pos))
        {
          cout  << "myid=" << myid<<"; c->getIncidCell(pos)="<<c->getIncidCell(pos)<<"; i="<<i<<"; pos="<<pos ;
          throw;
        };
        
        // verifica a face
        f = mesh->getFacetPtr(cell->getFacetId(i));
        int icf = f->getIncidCell();
        if(!(icf==myid || icf==cell->getIncidCell(i)))
        {
          cout <<"myid="<<myid<<"; icf="<<icf<<"; i="<<i<<"; cell->getIncidCell(i)="<<cell->getIncidCell(i)<<"\n";
          throw;
        }
        if (icf==myid)
        {
          if(f->getPosition() != i)
          {
            cout << "myid=" << myid<<"; f->getPosition()="<<f->getPosition()<<"; i="<<i<<"\n";
            throw;
          }
        }
        else
        {
          if(f->getPosition() != pos)
          {
            cout << "myid=" << myid<<"; f->getPosition()="<<f->getPosition()<<"; pos="<<pos<<"; i="<<i<<"\n";
            throw;
          }
        }
      }
      else // bordo
      {
        // verifica a face
        f = mesh->getFacetPtr(cell->getFacetId(i));
        int icf = f->getIncidCell();
        // só pode ser o myid, pq do outro lado não tem ngm
        if(icf!=myid)
        {
          cout << "icf = " << icf << ", myid = " << myid << endl;
          throw;
        }
        
        // verifica se os nós da face estão no contorno
        mesh->getFacetNodesId(f, f_nds);
        for (int j = 0; j < nnpf; ++j)
        {
          if(!mesh->inBoundary(mesh->getNodePtr(f_nds[j])))
          {
            cout << "node="<<f_nds[j]<<"; icell="<<mesh->getNodePtr(f_nds[j])->getIncidCell()
                << "; pos="<< mesh->getNodePtr(f_nds[j])->getPosition();
            throw;
          }
        }
        
      }
    }
    
  }
  
  std::vector<int> ics;
  std::vector<int> cc_ids;
  std::vector<int>::iterator it;
  for (point_iterator point = mesh->pointBegin(); point != mesh->pointEnd(); ++point)
  {
    point->getAllIncidences(ics);
    
    cc_ids.clear();

    int myid = point.index();
    
    for (int i = 0; i < (int)ics.size()/2; ++i)
    {
      int ic = ics.at(2*i);
      int pos = ics.at(2*i+1);
      Cell *c = mesh->getCellPtr(ic);
      if(c==NULL)
      {
        cout << "c = NULL" << endl;
        throw;
      }
      if(myid != c->getNodeId(pos))
      {
        cout << "ic = " << ic << "; pos = " << pos;
        throw;
      }
      cc_ids.push_back(c->getConnectedComponentId());
    }
    
    std::sort(cc_ids.begin(), cc_ids.end());
    it = unique (cc_ids.begin(), cc_ids.end());
    
    // checks if all incident cells have distinct connected component id
    if( std::distance(cc_ids.begin(), it) != (int)cc_ids.size())
    {
      cout << "MERDA DE std::distance\n";
      throw;
    }
    
  }
  
  
}

void AppCtx::getVecNormals(Vec const* Vec_x_1, Vec & Vec_normal_)
{

  VectorXi          facet_nodes(nodes_per_facet);
  MatrixXd          x_coefs(nodes_per_facet, dim);                // coordenadas nodais da célula
  MatrixXd          x_coefs_trans(dim, nodes_per_facet);
  Vector            X(dim), Xc(3);
  Vector            normal(dim);
  Tensor            F(dim,dim-1);
  VectorXi          map(n_dofs_u_per_facet);
  int               tag;
  int               tag_0, tag_1, tag_2;
  bool              contrib_0, contrib_1, contrib_2;
  bool              virtual_mesh;
  bool              is_surface, is_solid, is_cl, is_fsi, is_slv, is_boundary;
  int               sign_, is_fsiid, is_slvid;
  int               pts_id[15];


  if (Vec_x_1==NULL)
    virtual_mesh = false;
  else
    virtual_mesh = true;

  Assembly(Vec_normal_);
  VecSet(Vec_normal_,0);
  Assembly(Vec_normal_);

  // LOOP NAS FACES DO CONTORNO
  facet_iterator facet = mesh->facetBegin();
  facet_iterator facet_end = mesh->facetEnd();
  for (; facet != facet_end; ++facet)
  {
    tag = facet->getTag();

    is_surface = is_in(tag, interface_tags);
    is_solid   = is_in(tag, solid_tags);
    is_cl      = is_in(tag, triple_tags);
    is_fsi     = is_in(tag, flusoli_tags);
    is_slv     = is_in(tag, slipvel_tags);
    is_boundary= mesh->inBoundary(&*facet);

    if ( !(is_surface || is_solid || is_cl || is_boundary || is_fsi || is_slv) )
      continue;
    //cout << tag << "  ";
    contrib_0 = true;
    contrib_1 = true;
    contrib_2 = true;
    // the solid normal doesn't contribute to the triple normal
    if (is_solid)
    {
      mesh->getFacetNodesId(&*facet, pts_id);

      tag_0 = mesh->getNodePtr(pts_id[0])->getTag();
      contrib_0 = !is_in(tag_0, triple_tags);

      tag_1 = mesh->getNodePtr(pts_id[1])->getTag();
      contrib_1 = !is_in(tag_1, triple_tags);

      if (dim==3)
      {
        tag_2 = mesh->getNodePtr(pts_id[2])->getTag();
        contrib_2 = !is_in(tag_2, triple_tags);
      }
    }

    dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(map.data(), &*facet); //std::cout << map.transpose() << std::endl;

    mesh->getFacetNodesId(&*facet, facet_nodes.data()); //std::cout << facet_nodes.transpose() << std::endl;
    if (virtual_mesh)
      VecGetValues(*Vec_x_1, map.size(), map.data(), x_coefs.data());
    else
      mesh->getNodesCoords(facet_nodes.begin(), facet_nodes.end(), x_coefs.data());
    x_coefs_trans = x_coefs.transpose(); //std::cout << x_coefs_trans << std::endl;

    // fix orientation in the case where the gas phase isn't passive
    {
      sign_ = 1;
      cell_handler cc = mesh->getCell(facet->getIncidCell());
      cell_handler oc = mesh->getCell(cc->getIncidCell(facet->getPosition()));
      if ( oc.isValid()  )
        if ( oc->getTag() > cc->getTag() )
          sign_ = -1;
    }

    if (false) // método alternativo que serve para todos os métodos (normal consistente)
    {
      // find the normal
      for (int k = 0; k < nodes_per_facet; ++k)
      {
        //tag_other = mesh->getNodePtr(facet_nodes(k))->getTag();

        F   = x_coefs_trans * dLphi_nf[k];

        if (dim==2)
        {
          normal(0) = +F(1,0);
          normal(1) = -F(0,0);
        }
        else
        {
          normal = cross(F.col(0),F.col(1));
        }
        //normal.normalize();
        VecSetValues(Vec_normal_, dim, map.data()+k*dim, normal.data(), ADD_VALUES);

      } // nodes

    }
    else if (mesh_cell_type == TETRAHEDRON4) // facet = triangle3
    {
      F   = x_coefs_trans * dLphi_nf[0];
      Vector a(F.col(0)), b(F.col(1)-F.col(0)), c(F.col(1));

      normal = cross(F.col(0), F.col(1));
      normal *= sign_;

      // 0
      if (contrib_0)
      {
        X = normal;
        X /= a.dot(a) * c.dot(c);
        VecSetValues(Vec_normal_, dim, map.data()+0*dim, X.data(), ADD_VALUES);
      }

      // 1
      if (contrib_1)
      {
        X = normal;
        X /= a.dot(a) * b.dot(b);
        VecSetValues(Vec_normal_, dim, map.data()+1*dim, X.data(), ADD_VALUES);
      }

      // 2
      if (contrib_2)
      {
        X = normal;
        X /= b.dot(b) * c.dot(c);
        VecSetValues(Vec_normal_, dim, map.data()+2*dim, X.data(), ADD_VALUES);
      }

    }
    else if(mesh_cell_type == TETRAHEDRON10) // facet = triangle6
    {
      Vector x0(x_coefs_trans.col(0));
      Vector x1(x_coefs_trans.col(1));
      Vector x2(x_coefs_trans.col(2));
      Vector x3(x_coefs_trans.col(3));
      Vector x4(x_coefs_trans.col(4));
      Vector x5(x_coefs_trans.col(5));

      //edges
      double e03 = (x0-x3).squaredNorm();
      double e05 = (x0-x5).squaredNorm();
      double e13 = (x1-x3).squaredNorm();
      double e14 = (x1-x4).squaredNorm();
      double e24 = (x2-x4).squaredNorm();
      double e25 = (x2-x5).squaredNorm();
      double e34 = (x3-x4).squaredNorm();
      double e35 = (x3-x5).squaredNorm();
      double e45 = (x4-x5).squaredNorm();

      // dividi o triangulo em 4 mini-triangulos

      // node 0
      if (contrib_0)
      {
        cross(normal,x3-x0,x5-x0);
        normal /= e03*e05;
        normal *= sign_;
        VecSetValues(Vec_normal_, dim, map.data()+0*dim, normal.data(), ADD_VALUES);
      }

      // node 1
      if (contrib_1)
      {
        cross(normal,x4-x1,x3-x1);
        normal /= e14*e13;
        normal *= sign_;
        VecSetValues(Vec_normal_, dim, map.data()+1*dim, normal.data(), ADD_VALUES);
      }

      // node 2
      if (contrib_2)
      {
        cross(normal,x5-x2,x4-x2);
        normal /= e25*e24;
        normal *= sign_;
        VecSetValues(Vec_normal_, dim, map.data()+2*dim, normal.data(), ADD_VALUES);
      }

      // node 3
      if (contrib_0 && contrib_1)
      {
        normal = cross(x1-x3,x4-x3)/(e13*e34) + cross(x4-x3,x5-x3)/(e34*e35) + cross(x5-x3,x0-x3)/(e35*e03);
        normal *= sign_;
        VecSetValues(Vec_normal_, dim, map.data()+3*dim, normal.data(), ADD_VALUES);
      }

      // node 4
      if (contrib_1 && contrib_2)
      {
        normal = cross(x2-x4,x5-x4)/(e24*e45) + cross(x5-x4,x3-x4)/(e45*e34) + cross(x3-x4,x1-x4)/(e34*e14);
        normal *= sign_;
        VecSetValues(Vec_normal_, dim, map.data()+4*dim, normal.data(), ADD_VALUES);
      }

      // node 5
      if (contrib_2 && contrib_0)
      {
        normal = cross(x0-x5,x3-x5)/(e05*e35) + cross(x3-x5,x4-x5)/(e35*e45) + cross(x4-x5,x2-x5)/(e45*e25);
        normal *= sign_;
        VecSetValues(Vec_normal_, dim, map.data()+5*dim, normal.data(), ADD_VALUES);
      }
    }
    else if(mesh_cell_type == TRIANGLE3)
    {
      normal(0) = x_coefs_trans(1,1)-x_coefs_trans(1,0);
      normal(1) = x_coefs_trans(0,0)-x_coefs_trans(0,1);
      //normal /= (x_coefs_trans.col(0)-x_coefs_trans.col(1)).squaredNorm();//consistent normal
      normal.normalize();                                                   //mass conserving normal
      normal *= ((x_coefs_trans.col(0)-x_coefs_trans.col(1)).norm());     //mass conserving normal
      normal *= sign_;
      if (contrib_0) //contribution to node a
        VecSetValues(Vec_normal_, dim, map.data()+0*dim, normal.data(), ADD_VALUES); //map.data(): pointer to position 0,
      if (contrib_1) //contribution to node b                                        //map.data()+1*dim: pointer to position
        VecSetValues(Vec_normal_, dim, map.data()+1*dim, normal.data(), ADD_VALUES); //dim, so takes 0 and 1 first, then 2 and 3
      //if (is_surface)
      //cout << "normal = " << normal(0) << " " << normal(1) << endl;
    }
    else if(mesh_cell_type == TRIANGLE6) // dividi a aresta em duas partes
    {
      normal(0) = x_coefs_trans(1,2)-x_coefs_trans(1,0);
      normal(1) = x_coefs_trans(0,0)-x_coefs_trans(0,2);
      //normal /= (x_coefs_trans.col(0)-x_coefs_trans.col(2)).squaredNorm();
      normal.normalize();
      normal *= ((x_coefs_trans.col(0)-x_coefs_trans.col(2)).norm()/2);
      normal *= sign_;
      if (contrib_0)
        VecSetValues(Vec_normal_, dim, map.data()+0*dim, normal.data(), ADD_VALUES);
      VecSetValues(Vec_normal_, dim, map.data()+2*dim, normal.data(), ADD_VALUES);

      normal(0) = x_coefs_trans(1,1)-x_coefs_trans(1,2);
      normal(1) = x_coefs_trans(0,2)-x_coefs_trans(0,1);
      //normal /= (x_coefs_trans.col(1)-x_coefs_trans.col(2)).squaredNorm();
      normal.normalize();
      normal *= ((x_coefs_trans.col(1)-x_coefs_trans.col(2)).norm()/2);
      normal *= sign_;
      if (contrib_1)
        VecSetValues(Vec_normal_, dim, map.data()+1*dim, normal.data(), ADD_VALUES);
      VecSetValues(Vec_normal_, dim, map.data()+2*dim, normal.data(), ADD_VALUES);
    }

  }  //end for facet
  Assembly(Vec_normal_);

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    tag = point->getTag();

    is_surface = is_in(tag, interface_tags);
    is_solid   = is_in(tag, solid_tags);
    is_cl      = is_in(tag, triple_tags);
    is_fsiid   = is_in_id(tag, flusoli_tags);
    is_slv     = is_in(tag, slipvel_tags);
    is_fsi     = is_in(tag, flusoli_tags);
    is_slvid   = is_in_id(tag, slipvel_tags);
    is_fsiid   = is_in_id(tag, flusoli_tags);
    is_boundary= mesh->inBoundary(&*point);

    if ( !(is_surface || is_solid || is_cl || is_fsi || is_slv || is_boundary) )
      continue;

    getNodeDofs(&*point, DH_MESH, VAR_M, map.data());

    if (!is_in(tag, solid_tags))// && !is_in(tag, triple_tags))
    {
      VecGetValues(Vec_normal_, dim, map.data(),normal.data());
      normal.normalize();
      VecSetValues(Vec_normal_, dim, map.data(),normal.data(), INSERT_VALUES);
      Assembly(Vec_normal_);
    }
    else
    {
      if (virtual_mesh)
        VecGetValues(*Vec_x_1, dim, map.data(), X.data());
      else
        point->getCoord(X.data(),dim);
      normal = -solid_normal(X,current_time,tag);

      VecSetValues(Vec_normal_, dim, map.data(), normal.data(), INSERT_VALUES);
      Assembly(Vec_normal_);
    }
    if ((exact_normal || is_boundary) /*is_curvt*/ && (is_fsiid+is_slvid))
    {
      point->getCoord(X.data(),dim);
      Xc = XG_0[is_fsiid+is_slvid-1];  //TODO: see what's the correct XG_
      normal = exact_normal_ellipse(X,Xc,0.0/*theta_ini[is_fsiid+is_slvid-1]*/,RV[is_fsiid+is_slvid-1](0),RV[is_fsiid+is_slvid-1](1),dim); //theta_ini[is_fsiid+is_slvid-1]
      //cout << "HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE" << endl;
      VecSetValues(Vec_normal_, dim, map.data(), normal.data(), INSERT_VALUES);
      Assembly(Vec_normal_);
    }

  }
  Assembly(Vec_normal_);

}

//void AppCtx::getVecNormals(Vec const* Vec_normal_, Vec & Vec_tangent)
//{

//}

/// @param[out] Vec_normal_
/// @param[out] Vec_x_
void AppCtx::smoothsMesh(Vec & Vec_normal_, Vec &Vec_x_)
{
  //int        nodeid;
  //double     *Vec_x__array;
  bool const u_has_edge_assoc_dof = shape_phi_c->numDofsAssociatedToFacet()+shape_phi_c->numDofsAssociatedToCorner() > 0;

  Vector      Xm(dim); // X mean
  Vector      X0(dim);
  Vector      Xi(dim);
  Vector      dX(dim);
  Vector      normal(dim);
  Vector      tmp(dim), tmp2(dim);
  Vector      Uf(dim), Ue(dim), Umsh(dim); // Ue := elastic velocity
  int         tag;
  int         viz_tag;
  int         iVs[128], *iVs_end;
  int         iCs[128], viCs[128];
  VectorXi    vtx_dofs_umesh(dim);  // indices de onde pegar a velocidade
  VectorXi    vtx_dofs_fluid(dim); // indices de onde pegar a velocidade
  VectorXi    edge_dofs_umesh(3*dim);
  VectorXi    edge_dofs_fluid((2+u_has_edge_assoc_dof)*dim);
  VectorXi    edge_nodes(3);
  double      error;
  double      old_quality, new_quality;
  bool        in_boundary;
  //int        id;
  bool        is_surface;
  bool        is_solid;

  int const n_smooths = 10;

  /* suavização laplaciana */
  for (int smooth_it = 0; smooth_it < n_smooths; ++smooth_it)
  {
    error = 0;
    getVecNormals(&Vec_x_, Vec_normal_);

    // VERTICES
    point_iterator point = mesh->pointBegin();
    point_iterator point_end = mesh->pointEnd();
    for (; point != point_end; ++point)
    {
      tag = point->getTag();

      is_surface = is_in(tag, interface_tags);
      is_solid   = is_in(tag, solid_tags);

      in_boundary =  is_surface || is_solid;

      // pula o caso em que o ponto está no contoro mas não tem boundary smoothing
      if (!boundary_smoothing && in_boundary)
        continue;

      if (is_in(tag,triple_tags) || is_in(tag,feature_tags))
          continue;

      if (is_in(tag,dirichlet_tags) || is_in(tag,neumann_tags) || is_in(tag,periodic_tags))
          continue;

      if (mesh->isVertex(&*point))
      {
        //if (  is_in(tag,interface_tags) || is_in(tag,triple_tags) || is_in(tag,solid_tags) ||
        //    is_in(tag,dirichlet_tags) || is_in(tag,neumann_tags)  )

        dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), &*point);
        VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), X0.data()); // old coord

        Xm = Vector::Zero(dim);
        //iVs_end = mesh->connectedVtcs(&*point, iVs);
        iVs_end = mesh->connectedVtcs(&*point, iVs, iCs, viCs);

        if (!in_boundary)
        {
          int N=0;
          Point const* viz_pt;
          for (int *it = iVs; it != iVs_end ; ++it)
          {
            viz_pt = mesh->getNodePtr(*it);
            viz_tag = viz_pt->getTag();
            dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), viz_pt);
            VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data());
            ++N;
            Xm += tmp;
            //if (isFixedPoint(viz_tag))
            //{
            //  Xm += 5*X0;
            //  N += 5;
            //}
          }
          Xm /= N;

          dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), &*point);
          //// compute error
          VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data()); // old coord
          error += (tmp-Xm).norm();
          old_quality = getCellPatchQuality(Vec_x_, iCs);

          VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), Xm.data(), INSERT_VALUES);

          new_quality = getCellPatchQuality(Vec_x_, iCs);

          // se a qualidade piorou, volta no que estava antes
          if (new_quality < old_quality)
          {
            VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data(), INSERT_VALUES);
          }

        }
        else
        {
          int N=0;
          for (int *it = iVs; it != iVs_end ; ++it)
          {
            Point const* viz_pt = mesh->getNodePtr(*it);
            //if (viz_pt->getTag()!=tag && !is_in( viz_pt->getTag(), triple_tags ))
            viz_tag = viz_pt->getTag();
            if (viz_tag!=tag && (is_in(viz_tag,interface_tags) || is_in(viz_tag,solid_tags)) )
              continue;
            if (viz_tag!=tag && !is_in(viz_tag,triple_tags) && !isFixedPoint(viz_tag) && !is_in(viz_tag,feature_tags))
              continue;
            dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), mesh->getNodePtr(*it));
            // debug
            VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data());
            ++N;
            Xm += tmp;
            //if (isFixedPoint(viz_tag))
            //{
            //  Xm += 3*X0;
            //  N += 3;
            //}
          }

          dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), &*point);
          VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), Xi.data());

          if (dim==3)
            Xm = (N*Xi + 2*Xm)/(3*N);
            //Xm = Xm/N;
          else
            //Xm = (N*Xi + Xm)/(2*N);
            Xm = Xm/N;
          //

          dX = Xm - Xi;
          VecGetValues(Vec_normal_, dim, vtx_dofs_umesh.data(), normal.data());
          dX -= normal.dot(dX)*normal;
          Xi += dX;

          // compute error
          VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data());
          error += (tmp-Xi).norm();

          //old_quality = getCellPatchQuality(Vec_x_, iCs);
          VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), Xi.data(), INSERT_VALUES);
          //new_quality = getCellPatchQuality(Vec_x_, iCs);
          //// se a qualidade piorou, volta no que estava antes
          //if (new_quality < old_quality)
          //{
          //  VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data(), INSERT_VALUES);
          //}
        }

      }

    } // end point


    // MID NODES
    point = mesh->pointBegin();
    point_end = mesh->pointEnd();
    if (u_has_edge_assoc_dof)
    for (; point != point_end; ++point)
    {
      tag = point->getTag();

      is_surface = is_in(tag, interface_tags);
      is_solid   = is_in(tag, solid_tags);

      in_boundary = mesh->inBoundary(&*point) || is_surface || is_solid;

      // pula o caso em que o ponto está no contoro mas não tem boundary smoothing
      if (!boundary_smoothing && in_boundary)
        continue;

      if (!mesh->isVertex(&*point))
      {

        const int m = point->getPosition() - mesh->numVerticesPerCell();
        Cell const* icell = mesh->getCellPtr(point->getIncidCell());
        if (dim==3)
        {
          Corner *edge = mesh->getCornerPtr(icell->getCornerId(m));
          dof_handler[DH_MESH].getVariable(VAR_M).getCornerDofs(edge_dofs_umesh.data(), &*edge);
          dof_handler[DH_UNKM].getVariable(VAR_U).getCornerDofs(edge_dofs_fluid.data(), &*edge);
          mesh->getCornerNodesId(&*edge, edge_nodes.data());
        }
        else // dim=2
        {
          Facet *edge = mesh->getFacetPtr(icell->getFacetId(m));
          dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(edge_dofs_umesh.data(), &*edge);
          dof_handler[DH_UNKM].getVariable(VAR_U).getFacetDofs(edge_dofs_fluid.data(), &*edge);
          mesh->getFacetNodesId(&*edge, edge_nodes.data());
        }

        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data(), Xm.data());
        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data()+dim, tmp.data());
        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data()+2*dim, Xi.data());    // mid

        if (in_boundary)
        {
          Xm = (Xm+tmp+2*Xi)/4.;

          //Xm = (Xm+tmp)/2.;

          dX = Xm - Xi;
          VecGetValues(Vec_normal_, dim, edge_dofs_umesh.data()+2*dim, normal.data());
          dX -= normal.dot(dX)*normal;
          Xi += dX;
        }
        else
        {
          Xi = (Xm+tmp)/2.;
        }

        // compute error
        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data()+2*dim, tmp.data());
        error += (tmp-Xi).norm();

        VecSetValues(Vec_x_, dim, edge_dofs_umesh.data()+2*dim, Xi.data(), INSERT_VALUES);
        Assembly(Vec_x_);

      }

    } // end point


    error /= mesh->numNodes();
    //if (error < 1.e-10)
    //{
    //  cout << "STOP, " << smooth_it << " iterations\n";
    //  break;
    //}


  } // end smooth
  Assembly(Vec_x_);

  getVecNormals(&Vec_x_, Vec_normal_);
}

// copy mesh of the data structure to a Petsc Vector
void AppCtx::copyMesh2Vec(Vec & Vec_xmsh)
{
  //bool const u_has_edge_assoc_dof = shape_phi_c->numDofsAssociatedToFacet()+shape_phi_c->numDofsAssociatedToCorner() > 0;

  Vector     Xi(dim);
  VectorXi   node_dofs_mesh(dim);  // indices de onde pegar a velocidade

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    getNodeDofs(&*point, DH_MESH, VAR_M, node_dofs_mesh.data());
    point->getCoord(Xi.data(),dim); //std::cout << std::endl << Xi << std::endl << std::endl;
    VecSetValues(Vec_xmsh, dim, node_dofs_mesh.data(), Xi.data(), INSERT_VALUES);
  }
  Assembly(Vec_xmsh); //VecView(Vec_xmsh,PETSC_VIEWER_STDOUT_WORLD);
}

// copy mesh of the data structure to a Petsc Vector
void AppCtx::copyVec2Mesh(Vec const& Vec_xmsh)
{
  //bool const u_has_edge_assoc_dof = shape_phi_c->numDofsAssociatedToFacet()+shape_phi_c->numDofsAssociatedToCorner() > 0;

  Vector     Xi(dim);
  VectorXi   node_dofs_mesh(dim);  // indices de onde pegar a velocidade

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    getNodeDofs(&*point, DH_MESH, VAR_M, node_dofs_mesh.data());
    VecGetValues(Vec_xmsh, dim, node_dofs_mesh.data(), Xi.data());
    point->setCoord(Xi.data(),dim);
  }
}

// copy mesh of the data structure to a Petsc Vector
void AppCtx::swapMeshWithVec(Vec & Vec_xmsh)
{
  //double     *Vec_xmsh_array;
  //bool const u_has_edge_assoc_dof = shape_phi_c->numDofsAssociatedToFacet()+shape_phi_c->numDofsAssociatedToCorner() > 0;

  Vector     tmp(dim); // X mean
  Vector     Xi(dim); // X mean
  VectorXi   node_dofs_umesh(dim);  // indices de onde pegar a velocidade

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    getNodeDofs(&*point, DH_MESH, VAR_M, node_dofs_umesh.data());

    point->getCoord(tmp.data(),dim);
    VecGetValues(Vec_xmsh, dim, node_dofs_umesh.data(), Xi.data());

    point->setCoord(Xi.data(),dim);
    VecSetValues(Vec_xmsh, dim, node_dofs_umesh.data(), tmp.data(), INSERT_VALUES);
  }

  Assembly(Vec_xmsh);
}

PetscErrorCode AppCtx::meshAdapt()
{
  //PetscFunctionReturn(0);
  PetscErrorCode      ierr;

  // only for 2d ... TODO for 3d too
  if (dim != 2)
    PetscFunctionReturn(0);
  // only for linear elements
  if (mesh->numNodesPerCell() > mesh->numVerticesPerCell())
    PetscFunctionReturn(0);

  const Real TOL = 0.6; //0.6

  typedef tuple<int, int, int> EdgeVtcs; // get<0> = mid node, get<1> = top node, get<2> = bot node


  std::list<EdgeVtcs> adde_vtcs;

  CellElement *edge;
  Real h;
  Real expected_h;
  VectorXi edge_nodes(3); // 3 nodes at most
  Vector Xa(dim), Xb(dim);
  int tag_a;
  int tag_b;
  int tag_e;

  bool mesh_was_changed = false;
  //cout << "#z=" << n_unknowns_fs << " #u=" << n_unknowns_u << " #v=" << n_dofs_v_mesh  << endl;
  //printf("ENTRANDO NO LOOP is_splitting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  // first make collapses, then make splittings.
  // NOTE: mark added points
  for (int is_splitting = 0; is_splitting < 2 ; ++is_splitting)
  {
    int const n_edges_total = (dim==2) ? mesh->numFacetsTotal() : mesh->numCornersTotal();

    //printf("ENTRANDO NO LOOP eid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    for (int eid = 0; eid < n_edges_total; ++eid)
    {

      if (dim == 2)
        edge = mesh->getFacetPtr(eid);
      else
        edge = mesh->getCornerPtr(eid);

      if (edge==NULL || edge->isDisabled())
        continue;

      if (dim == 2)
        mesh->getFacetNodesId(eid, edge_nodes.data());
      else
        mesh->getCornerNodesId(eid, edge_nodes.data());

      tag_a = mesh->getNodePtr(edge_nodes[0])->getTag();
      tag_b = mesh->getNodePtr(edge_nodes[1])->getTag();
      tag_e = edge->getTag();

      //if (is_in(tag_e, dirichlet_tags)||
      //    is_in(tag_e, neumann_tags) ||
      //    is_in(tag_e, interface_tags) ||
      //    is_in(tag_e, dirichlet_tags) ||
      //    is_in(tag_e, solid_tags) ||
      //    is_in(tag_e, periodic_tags) ||
      //    is_in(tag_e, feature_tags)  )
      //  continue;
      if (is_in(tag_e, flusoli_tags))
        continue;
      if (!( tag_a==tag_b && tag_b==tag_e ) && (is_splitting==0))  //only edges inside the same type of domain
        continue;                                                  //(ex: inside fluid)

      Point const* pt_a = mesh->getNodePtr(edge_nodes[0]);
      Point const* pt_b = mesh->getNodePtr(edge_nodes[1]);
      //cout << edge_nodes.transpose() << endl;
      if (pt_a->isMarked() || pt_b->isMarked())
        continue;

      pt_a->getCoord(Xa.data(),dim);
      pt_b->getCoord(Xb.data(),dim);
      
      h = (Xa-Xb).norm();

      expected_h = .5*(mesh_sizes[edge_nodes[0]] + mesh_sizes[edge_nodes[1]]);
      //cout << h << "  " << expected_h << "     " << mesh_sizes[edge_nodes[0]] << "  " << mesh_sizes[edge_nodes[1]] << endl;
      //printf("mesh size  %lf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", (h - expected_h)/expected_h);

      if (is_splitting )//&& (time_step%2==0))
      {
        if ((h - expected_h)/expected_h > TOL)
        {
          mesh_was_changed = true;
          //cout << mesh->getPointId(pt_a) << " " << mesh->getPointId(pt_b) << endl;
          //cout << h << "  " << expected_h << "     " << mesh_sizes[edge_nodes[0]] << "  " << mesh_sizes[edge_nodes[1]] << endl;
          int pt_id = MeshToolsTri::insertVertexOnEdge(edge->getIncidCell(), edge->getPosition(), 0.5, &*mesh);
          printf("INSERTED %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", pt_id);
          adde_vtcs.push_back(make_tuple(pt_id, edge_nodes[0], edge_nodes[1]));
          mesh->getNodePtr(pt_id)->setMarkedTo(true);
          if (pt_id < (int)mesh_sizes.size())
            mesh_sizes[pt_id] = expected_h;
          else
          {
            if (pt_id > (int)mesh_sizes.size())
            {
              printf("ERROR: Something with mesh_sizes is wrong!!\n");
              throw;
            }
            mesh_sizes.push_back(expected_h);
          }
        }
      }
      else if(!is_splitting )//&& !(time_step%2==0)) // is collapsing
      {
        if ((h - expected_h)/expected_h < -TOL)
        {
          mesh_was_changed = true;
          printf("COLLAPSED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
          printf("COLLAPSING  tags = %d  %d %d #################################################\n", tag_a, tag_b, tag_e);
          int pt_id = MeshToolsTri::collapseEdge2d(edge->getIncidCell(), edge->getPosition(), 0.0, &*mesh);
          printf("COLLAPSED %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", pt_id);
          //mesh->getNodePtr(pt_id)->setMarkedTo(true);
        }
      }

    } // end n_edges_total
  }
  //cout << "ANTES CHANGED" << endl;
  if (!mesh_was_changed)
    PetscFunctionReturn(0);
  //cout << "Y DESPUES CHANGED" << endl;
  // atualiza tudo!
  meshAliasesUpdate();
  //snes_fs = NULL;
  //ierr = SNESDestroy(&snes_fs);    CHKERRQ(ierr);//SNESReset(snes_fs);
  //snes_fs = NULL; snes_m = NULL;
  //ierr = SNESDestroy(&snes_m);     CHKERRQ(ierr);
  //snes_m = NULL;
  // free petsc matrices first and residues to save memory.
  // destroy only those who not depends on DofHandler
  //Destroy(Mat_Jac_m);
  //ierr = VecDestroy(&Vec_res_m);  CHKERRQ(ierr);
  //ierr = VecDestroy(&Vec_normal);  CHKERRQ(ierr);
  //ierr = VecDestroy(&Vec_v_mid);  CHKERRQ(ierr);
  //SNESReset(snes_m);
  //KSPReset(ksp_m);
  //PCReset(pc_m);
  //SNESLineSearchReset(linesearch);
  //Destroy(Mat_Jac_fs);  //Destroy(Mat_Jac);
  //ierr = VecDestroy(&Vec_res_fs);  CHKERRQ(ierr);  //Destroy(Vec_res);
  //SNESReset(snes_fs);  //SNESDestroy(&snes_fs); //SNESReset(snes);  //error con el SNESReset de snes_fs
  //KSPReset(ksp_fs);  //KSPReset(ksp);
  //PCReset(pc_fs);  //PCReset(pc);
  Destroy(Mat_Jac_fs);
  Destroy(Mat_Jac_m);
  Destroy(Vec_res_fs);
  Destroy(Vec_res_m);
  Destroy(Vec_normal);
  Destroy(Vec_v_mid);
  SNESReset(snes_fs);
  SNESReset(snes_m);
  KSPReset(ksp_fs);
  KSPReset(ksp_m);
  PCReset(pc_fs);
  PCReset(pc_m);
  SNESLineSearchReset(linesearch);

  DofHandler  dof_handler_tmp[2];
  
  // tranfers variables values from old to new mesh
  {
    // First fix the u-p unknows
    dof_handler_tmp[DH_MESH].copy(dof_handler[DH_MESH]);
    dof_handler_tmp[DH_UNKM].copy(dof_handler[DH_UNKM]);  //dof_handler_tmp[DH_UNKS].copy(dof_handler[DH_UNKS]);

    dofsUpdate();  //updates DH_ information: # variables u, z, p, mesh can change
    cout << "#z=" << n_unknowns_fs << " #u=" << n_unknowns_u << " #v=" << n_dofs_v_mesh  << endl;

    int n_solid_nodes_0 = dof_handler_tmp[DH_UNKM].getVariable(VAR_Z).numPositiveDofs()/3;
    int n_solid_nodes_1 = dof_handler    [DH_UNKM].getVariable(VAR_Z).numPositiveDofs()/3;
    int p_id_cor_0 = 3*(n_solid_nodes_0-n_solids);
    int p_id_cor_1 = 3*(n_solid_nodes_1-n_solids);
    //cout << n_solid_nodes_0 << " " << n_solid_nodes_1 << endl;

    Vec *petsc_vecs[] = {&Vec_ups_0, &Vec_ups_1, &Vec_x_0, &Vec_x_1}; //Vec *petsc_vecs[] = {&Vec_up_0, &Vec_up_1, &Vec_x_0, &Vec_x_1};
    int DH_t[]        = {DH_UNKM, DH_UNKM, DH_MESH, DH_MESH}; //int DH_t[]       = {DH_UNKS , DH_UNKS , DH_MESH, DH_MESH};
    int n_unks_t[]    = {n_unknowns_fs, n_unknowns_fs, n_dofs_v_mesh, n_dofs_v_mesh};
    //cout << n_unknowns_fs << " " << n_unknowns_u << " " << n_dofs_v_mesh << endl;
    std::vector<Real> temp;
    //cout << static_cast<int>( sizeof(DH_t)/sizeof(int) ) << endl;  // es 4 = size(DH_t)
    // NOTE: the mesh must not be changed in this loop
    for (int v = 0; v < static_cast<int>( sizeof(DH_t)/sizeof(int) ); ++v)
    {
      int vsize;
      //cout << v << "  ";
      VecGetSize(*petsc_vecs[v], &vsize);  //size of Vec_ups_0, Vec_ups_1, Vec_x_0, Vec_x_1
      //cout << vsize << endl;
      temp.assign(vsize, 0.0);

      double *array;

      // copy
      VecGetArray(*petsc_vecs[v], &array);
      for (int i = 0; i < vsize; ++i)
        temp[i] = array[i];
      VecRestoreArray(*petsc_vecs[v], &array);
      //*petsc_vecs[v] = NULL;
      Destroy(*petsc_vecs[v]);  //ierr = VecDestroy(&*petsc_vecs[v]);  CHKERRQ(ierr);
      //*petsc_vecs[v] = NULL;
      ierr = VecCreate(PETSC_COMM_WORLD, petsc_vecs[v]);                              CHKERRQ(ierr);
      ierr = VecSetSizes(*petsc_vecs[v], PETSC_DECIDE, n_unks_t[v]);                  CHKERRQ(ierr);  //dof_handler[DH_t[v]].numDofs()
      ierr = VecSetFromOptions(*petsc_vecs[v]);                                       CHKERRQ(ierr);
      if (v == 0 || v == 1){//DH_UNKM
        ierr = VecSetOption(*petsc_vecs[v], VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);
      }

      int tagP, nod_id;

      int dofs_0[64]; // old
      int dofs_1[64]; // new
      //cout << dof_handler_tmp[DH_t[v+2]].numVars() << endl;
      // copy data from old mesh to new mesh
      VecGetArray(*petsc_vecs[v], &array);
      for (point_iterator point = mesh->pointBegin(), point_end = mesh->pointEnd(); point != point_end; ++point)
      {
        if (!mesh->isVertex(&*point))
          continue;

        tagP = point->getTag();
        nod_id = is_in_id(tagP,flusoli_tags);

        if (!point->isMarked())
        {
          for (int k = 0; k < dof_handler_tmp[DH_t[v]].numVars(); ++k)  //ranges over # variables = 3 = u,z,p or 1 = mesh
          {
            if (v == 0 || v == 1){//DH_UNKM
              if (nod_id){
                if(k == VAR_Z){
                  //printf("v=%d,  point=%d, k=%d, nod_id=%d\n", v, mesh->getPointId(&*point), k, nod_id);
                  for (int l = 0; l < 3; l++){  // the 3 here is for Z quantity of Dofs for 2D case
                    dofs_0[l] = dof_handler_tmp[DH_t[v]].getVariable(VAR_U).numPositiveDofs() - 1
                              + 3*nod_id - 2 + l;
                    dofs_1[l] = dof_handler    [DH_t[v]].getVariable(VAR_U).numPositiveDofs() - 1
                              + 3*nod_id - 2 + l;
                  }
                  for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
                    array[dofs_1[j]] = temp.at(dofs_0[j]);
                }
                else if (k == VAR_P){
                  //printf("v=%d,  point=%d, k=%d, nod_id=%d\n", v, mesh->getPointId(&*point), k, nod_id);
                  dof_handler_tmp[DH_t[v]].getVariable(k).getVertexDofs(dofs_0, mesh->getPointId(&*point));  //cout << dofs_0[0] << " " << dofs_0[1] << " "  << dofs_0[2] << " "  << dofs_0[3] << endl;
                  dof_handler    [DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);          //cout << dofs_1[0] << " " << dofs_1[1] << " "  << dofs_1[2] << " "  << dofs_1[3] << endl;
                  dofs_0[0] = dofs_0[0] - p_id_cor_0;
                  dofs_1[0] = dofs_1[0] - p_id_cor_1;
                  for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
                    array[dofs_1[j]] = temp.at(dofs_0[j]);
                }
              }
              else{
                if (k == VAR_U){
                  //printf("v=%d,  point=%d, k=%d, nod_id=%d\n", v, mesh->getPointId(&*point), k, nod_id);
                  //dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, &*point);
                  dof_handler_tmp[DH_t[v]].getVariable(k).getVertexDofs(dofs_0, mesh->getPointId(&*point));  //cout << dofs_0[0] << " " << dofs_0[1] << " "  << dofs_0[2] << " "  << dofs_0[3] << endl;
                  dof_handler    [DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);          //cout << dofs_1[0] << " " << dofs_1[1] << " "  << dofs_1[2] << " "  << dofs_1[3] << endl;
                  for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
                    array[dofs_1[j]] = temp.at(dofs_0[j]);
                }
                else if (k == VAR_P){
                  //printf("v=%d,  point=%d, k=%d, nod_id=%d\n", v, mesh->getPointId(&*point), k, nod_id);
                  dof_handler_tmp[DH_t[v]].getVariable(k).getVertexDofs(dofs_0, mesh->getPointId(&*point));  //cout << dofs_0[0] << " " << dofs_0[1] << " "  << dofs_0[2] << " "  << dofs_0[3] << endl;
                  dof_handler    [DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);          //cout << dofs_1[0] << " " << dofs_1[1] << " "  << dofs_1[2] << " "  << dofs_1[3] << endl;
                  dofs_0[0] = dofs_0[0] - p_id_cor_0;
                  dofs_1[0] = dofs_1[0] - p_id_cor_1;
                  for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
                    array[dofs_1[j]] = temp.at(dofs_0[j]);
                }
              }
            }
            else if (v == 2 || v == 3){//DH_MESH
              //printf("v=%d,  point=%d, k=%d, nod_id=%d\n", v, mesh->getPointId(&*point), k, nod_id);
              //dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, &*point);
              dof_handler_tmp[DH_t[v]].getVariable(k).getVertexDofs(dofs_0, mesh->getPointId(&*point));  //cout << dofs_0[0] << " " << dofs_0[1] << " "  << dofs_0[2] << " "  << dofs_0[3] << endl;
              dof_handler    [DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);          //cout << dofs_1[0] << " " << dofs_1[1] << " "  << dofs_1[2] << " "  << dofs_1[3] << endl;
              for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
                array[dofs_1[j]] = temp.at(dofs_0[j]);
            }//end if value v
          }//end for k
        }//end if !marked
      }//end for point
      // interpolate values at new points
      std::list<EdgeVtcs>::iterator it     = adde_vtcs.begin();
      std::list<EdgeVtcs>::iterator it_end = adde_vtcs.end();
      for (; it != it_end; ++it)
      {
        int const pt_id        =  get<0>(*it);
        int const a_id         =  get<1>(*it);
        int const b_id         =  get<2>(*it);
        Point      * point = mesh->getNodePtr(pt_id);
        Point const* pt_a  = mesh->getNodePtr(a_id);
        Point const* pt_b  = mesh->getNodePtr(b_id);
        Point const* link[] = {pt_a, pt_b};
        int const Nlinks = sizeof(link)/sizeof(Point*);
        double const weight = 1./Nlinks;
        //printf("a_id = b_id %d %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", a_id, b_id);
        tagP = point->getTag();
        nod_id = is_in_id(tagP,flusoli_tags);
        for (int k = 0; k < dof_handler_tmp[DH_t[v]].numVars(); ++k)
        {
          if (v == 0 || v == 1){//DH_UNKM
            if(nod_id){
              if (k == VAR_Z){
                cout << "FALTA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!mesh 1063" << endl; //TODO: actualizar
              }
              else if (k == VAR_P){
                dof_handler[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
                for (int j = 0; j < Nlinks; ++j)
                {
                  dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, link[j]);
                  dofs_0[0] = dofs_0[0] - p_id_cor_0;
                  dofs_1[0] = dofs_1[0] - p_id_cor_1;
                  //printf("v = %d  x = %lf  y = %lf !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", v, temp[dofs_0[0]], temp[dofs_0[1]]);
                  for (int c = 0; c < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++c)
                    array[dofs_1[c]] += weight*temp[dofs_0[c]];
                }
              }
            }
            else{
              if (k == VAR_U){
                dof_handler[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
                for (int j = 0; j < Nlinks; ++j)
                {
                  dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, link[j]);
                  //printf("v = %d  x = %lf  y = %lf !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", v, temp[dofs_0[0]], temp[dofs_0[1]]);
                  for (int c = 0; c < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++c)
                    array[dofs_1[c]] += weight*temp[dofs_0[c]];
                }
              }
              else if(k == VAR_P){
                dof_handler[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
                for (int j = 0; j < Nlinks; ++j)
                {
                  dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, link[j]);
                  dofs_0[0] = dofs_0[0] - p_id_cor_0;
                  dofs_1[0] = dofs_1[0] - p_id_cor_1;
                  //printf("v = %d  x = %lf  y = %lf !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", v, temp[dofs_0[0]], temp[dofs_0[1]]);
                  for (int c = 0; c < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++c)
                    array[dofs_1[c]] += weight*temp[dofs_0[c]];
                }
              }
            }
          }//end DH_UNKM
          else if (v == 2 || v == 3){//DH_MESH
            dof_handler[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
            for (int j = 0; j < Nlinks; ++j)
            {
              dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, link[j]);
              //printf("v = %d  x = %lf  y = %lf !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", v, temp[dofs_0[0]], temp[dofs_0[1]]);
              for (int c = 0; c < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++c)
                array[dofs_1[c]] += weight*temp[dofs_0[c]];
            }
          }
        } //end DH_MESH
      } //end for k variables
      VecRestoreArray(*petsc_vecs[v], &array);
      Assembly(*petsc_vecs[v]);
    } // end loop in vectors
    //Destroy(Vec_ups_1);
    std::list<EdgeVtcs>::iterator it     = adde_vtcs.begin();
    std::list<EdgeVtcs>::iterator it_end = adde_vtcs.end();
    for (; it != it_end; ++it)
    {
      int const pt_id =  get<0>(*it);
      mesh->getNodePtr(pt_id)->setMarkedTo(false);
    }

    //for (point_iterator point = mesh->pointBegin(), point_end = mesh->pointEnd(); point != point_end; ++point)
    //{
    //  if (point->isMarked())
    //  {
    //    printf("%d MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA\n", mesh->getPointId(&*point));
    //  }
    //}


  } // end tranf

  //cout << "end v cycle" << endl;
  //Vec Vec_res;
//  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res);                     CHKERRQ(ierr);
//  ierr = VecSetSizes(Vec_res, PETSC_DECIDE, n_unknowns);            CHKERRQ(ierr);
//  ierr = VecSetFromOptions(Vec_res);                                CHKERRQ(ierr);

  //Vec Vec_res;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_fs);                     CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_res_fs, PETSC_DECIDE, n_unknowns_fs);            CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_res_fs);                                CHKERRQ(ierr);

  //Vec Vec_v_mid
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_v_mid);                  CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_v_mid, PETSC_DECIDE, n_dofs_v_mesh);      CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_v_mid);                             CHKERRQ(ierr);

  //Vec Vec_normal;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_normal);                  CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_normal, PETSC_DECIDE, n_dofs_v_mesh);      CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_normal);                             CHKERRQ(ierr);

  //Vec Vec_res_m;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_m);                     CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_res_m, PETSC_DECIDE, n_dofs_v_mesh);         CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_res_m);

  std::vector<int> nnz;
  //if(false)
//  {
//    nnz.resize(n_unknowns,0);
//    std::vector<SetVector<int> > table;
//    dof_handler[DH_UNKS].getSparsityTable(table); // TODO: melhorar desempenho, função mt lenta
    //FEP_PRAGMA_OMP(parallel for)
//    for (int i = 0; i < n_unknowns; ++i)
//      nnz[i] = table[i].size();
//  }

  //Mat Mat_Jac;
//  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac);                                      CHKERRQ(ierr);
//  ierr = MatSetSizes(Mat_Jac, PETSC_DECIDE, PETSC_DECIDE, n_unknowns, n_unknowns);   CHKERRQ(ierr);
//  ierr = MatSetFromOptions(Mat_Jac);                                                 CHKERRQ(ierr);
//  ierr = MatSeqAIJSetPreallocation(Mat_Jac,  0, nnz.data());                         CHKERRQ(ierr);
//  ierr = MatSetOption(Mat_Jac,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);           CHKERRQ(ierr);

  //Mat Mat_Jac;
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_fs);                                      CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_fs, PETSC_DECIDE, PETSC_DECIDE, n_unknowns_fs, n_unknowns_fs);   CHKERRQ(ierr);
  ierr = MatSetFromOptions(Mat_Jac_fs);                                                 CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_fs, PETSC_DEFAULT, NULL);                    CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_fs,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);           CHKERRQ(ierr);

  // mesh
  nnz.clear();
  int n_mesh_dofs = dof_handler[DH_MESH].numDofs();
  {
    std::vector<SetVector<int> > table;
    dof_handler[DH_MESH].getSparsityTable(table); // TODO: melhorar desempenho, função mt lenta

    nnz.resize(n_mesh_dofs, 0);

    //FEP_PRAGMA_OMP(parallel for)
    for (int i = 0; i < n_mesh_dofs; ++i)
      nnz[i] = table[i].size();

  }
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_m);                                        CHKERRQ(ierr);
  ierr = MatSetType(Mat_Jac_m,MATSEQAIJ);                                                CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_m, PETSC_DECIDE, PETSC_DECIDE, n_mesh_dofs, n_mesh_dofs);   CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_m,  0, nnz.data());                           CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);             CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_SYMMETRIC,PETSC_TRUE);                               CHKERRQ(ierr);

//  ierr = SNESSetFunction(snes, Vec_res, FormFunction, this);                             CHKERRQ(ierr);
//  ierr = SNESSetJacobian(snes, Mat_Jac, Mat_Jac, FormJacobian, this);                    CHKERRQ(ierr);
//  ierr = SNESSetConvergenceTest(snes,CheckSnesConvergence,this,PETSC_NULL);              CHKERRQ(ierr);
//  ierr = SNESGetKSP(snes,&ksp);                                                          CHKERRQ(ierr);
//  ierr = KSPGetPC(ksp,&pc);                                                              CHKERRQ(ierr);
//  ierr = KSPSetOperators(ksp,Mat_Jac,Mat_Jac,SAME_NONZERO_PATTERN);                      CHKERRQ(ierr);
//  ierr = SNESSetFromOptions(snes);                                                       CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fs);
  ierr = SNESSetFunction(snes_fs, Vec_res_fs, FormFunction_fs, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_fs, Mat_Jac_fs, Mat_Jac_fs, FormJacobian_fs, this);        CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(snes_fs,CheckSnesConvergence,this,PETSC_NULL);           CHKERRQ(ierr);
  ierr = SNESGetKSP(snes_fs,&ksp_fs);                                                    CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_fs,&pc_fs);                                                        CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_fs,Mat_Jac_fs,Mat_Jac_fs,SAME_NONZERO_PATTERN);             CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_fs);                                                    CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_m);
  ierr = SNESSetFunction(snes_m, Vec_res_m, FormFunction_mesh, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_m, Mat_Jac_m, Mat_Jac_m, FormJacobian_mesh, this);         CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes_m,&linesearch);                                          CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);                          CHKERRQ(ierr);
  ierr = SNESGetKSP(snes_m,&ksp_m);                                                      CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_m,&pc_m);                                                          CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_m,Mat_Jac_m,Mat_Jac_m,SAME_NONZERO_PATTERN);                CHKERRQ(ierr);
  //ierr = KSPSetType(ksp_m,KSPCG);                                                    CHKERRQ(ierr);
  //ierr = PCSetType(pc_m,PCILU);                                                      CHKERRQ(ierr);

  if(!nonlinear_elasticity)
  {
    ierr = SNESSetType(snes_m, SNESKSPONLY); CHKERRQ(ierr);
  }
  ierr = SNESSetFromOptions(snes_m); CHKERRQ(ierr);

  // check
  if (false)
  {
    int const n_edges_total = (dim==2) ? mesh->numFacetsTotal() : mesh->numCornersTotal();
    
    //printf("ENTRANDO NO LOOP eid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    for (int eid = 0; eid < n_edges_total; ++eid)
    {

      if (dim == 2)
        edge = mesh->getFacetPtr(eid);
      else
        edge = mesh->getCornerPtr(eid);

      if (edge==NULL || edge->isDisabled())
        continue;

      int tag = edge->getTag();

      if (tag != 2 && tag != 3 && tag != 7)
      {
        printf(" MERDA  MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA\n");
        printf("%d\n", tag);
        throw;
      }
      
      if (tag != 2 && tag != 3 && mesh->inBoundary((Facet*)edge))
      {
        printf(" MERDA  MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA\n");
        printf("%d\n", tag);
        throw;
      }
      
    }    
    
  }

  checkConsistencyTri(&*mesh);

  printf("\nMesh adapted.\n\n");

  PetscFunctionReturn(0);
}

//PetscErrorCode AppCtx::calcMeshVelocity(Vec const& Vec_x_0, Vec const& Vec_x_1, Vec &Vec_v_mid)
//{
//  PetscErrorCode ierr;
//  ierr = VecCopy(Vec_x_1, Vec_v_mid);      CHKERRQ(ierr);
//  ierr = VecAXPY(Vec_v_mid,-1.,Vec_x_0);   CHKERRQ(ierr);
//  ierr = VecScale(Vec_v_mid, 1./dt);       CHKERRQ(ierr);
//  Assembly(Vec_v_mid);
//
//  // DEBUG
//  //Vector      U0(dim);
//  //Vector      X0(dim);
//  //Vector      X1(dim);
//  //VectorXi    node_dofs_fluid(dim);
//  //getNodeDofs(&*mesh->getNodePtr(120), DH_UNKS, VAR_U, node_dofs_fluid.data());
//  //VecGetValues(Vec_v_mid,  dim, node_dofs_fluid.data(), U0.data());
//  //VecGetValues(Vec_x_0,  dim, node_dofs_fluid.data(), X0.data());
//  //VecGetValues(Vec_x_1,  dim, node_dofs_fluid.data(), X1.data());
//  //cout << "VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY = " << U0(0) << " " << U0(1) << endl;
//  //cout << "VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY = " << X0(0) << " " << X0(1) << endl;
//  //cout << "VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY VELOCITY = " << X1(0) << " " << X1(1) << endl;
//  //cout.flush();
//
//
//  PetscFunctionReturn(0);
//}
//

// by elasticity
// @brief A mean between Vec_up_0 and Vec_up_1 is used as boundary conditions to an elasticity problem. This mean
// is computed by = vtheta*Vec_up_1 + (1-Vec_up_1)*Vec_up_0

PetscErrorCode AppCtx::calcMeshVelocity(Vec const& Vec_x_0, Vec const& Vec_up_0, Vec const& Vec_up_1, double vtheta, Vec &Vec_v_mid, double tt)
{
  PetscErrorCode ierr;

  // The normal used must be at time step n.

  // set boundary conditions on the initial guess
  {
    VectorXi    node_dofs_mesh(dim);
    VectorXi    node_dofs_fluid_fs(dim);
    VectorXi    node_dofs_solid_fs(LZ);

    Vector      Xp(dim), Xg(dim), Vels(dim), Vm(3);
    Vector3d    Xgt;
    int         tag, nod_id, nod_is, nod_vs, nodsum;

    Vector      U0(dim), U0_fs(LZ);
    Vector      X0(dim), Y0(dim);
    Vector      U1(dim), U1_fs(LZ);
    Vector      tmp(dim), tmp_fs(LZ), tmp_n(dim);
    Vector      k1(dim),k2(dim),k3(dim),k4(dim); //  RK4

    std::vector<bool>   SV(n_solids,false);  //solid visited history

    point_iterator point = mesh->pointBegin();
    point_iterator point_end = mesh->pointEnd();
    for (; point != point_end; ++point)  //to calculate Vec_v_mid at each point (initial guess)
    {
      tag = point->getTag();
      nod_id = is_in_id(tag,flusoli_tags);
      nod_is = is_in_id(tag,solidonly_tags);
      nod_vs = is_in_id(tag,slipvel_tags);
      nodsum = nod_id+nod_is+nod_vs;

      getNodeDofs(&*point, DH_MESH, VAR_M, node_dofs_mesh.data());
      getNodeDofs(&*point, DH_UNKM, VAR_U, node_dofs_fluid_fs.data());

      if (force_mesh_velocity)
      {
        VecGetValues(Vec_x_0, dim, node_dofs_mesh.data(), X0.data());  //node mesh vel value
        if (is_sfip && is_slipv) VecCopy(Vec_slipv_0, Vec_slipv_1);
        //k1 = v_exact(X0,tt,tag);
        //k2 = v_exact(X0+0.5*k1*dt,tt+0.5*dt,tag);
        //k3 = v_exact(X0+0.5*k2*dt,tt+0.5*dt,tag);
        //k4 = v_exact(X0+k3*dt,tt+dt,tag);
        //tmp =  (k1 + 2.*(k2+k3) + k4)/6.; // velocity
        
        if (unsteady){
          int const N = 16;
          double Dt = dt/N, TT = tt;
          Y0 = X0;
          tmp.setZero();
          for (int jj = 0; jj < N ; ++jj)
          {
            k1 = v_exact(Y0,TT,tag);
            k2 = v_exact(Y0+0.5*k1*Dt,TT+0.5*Dt,tag);
            k3 = v_exact(Y0+0.5*k2*Dt,TT+0.5*Dt,tag);
            k4 = v_exact(Y0+k3*Dt,TT+Dt,tag);
            tmp =  (k1 + 2.*(k2+k3) + k4)/6.; // velocity
            Y0 += Dt*tmp;
            TT += Dt;
          }
          //if (is_bdf2 && !is_bdf_cte_vel && time_step !=0)
          if (is_bdf2)
          {
            if (time_step == 0)
            {
              if (is_bdf_euler_start)
                tmp = v_exact(Y0, tt+dt, tag);
            }
            else
            {
              if (!is_bdf_cte_vel)
                tmp = v_exact(Y0, tt+dt, tag);
              else
                tmp = (Y0 - X0)/dt;
            }

          }
          else
            tmp = (Y0 - X0)/dt;
        }
        else{
          tmp = v_exact(Y0, 0.0, tag);
        }

        VecSetValues(Vec_v_mid, dim, node_dofs_mesh.data(), tmp.data(), INSERT_VALUES);
      }  //if for RK4  //end if force_mesh_velocity
      else
      {
        if ( true && (  is_in(tag, neumann_tags) || is_in(tag, dirichlet_tags) || is_in(tag, periodic_tags) || is_in(tag, feature_tags)  )   )
        {
          tmp.setZero();
          VecSetValues(Vec_v_mid, dim, node_dofs_mesh.data(), tmp.data(), INSERT_VALUES);
        }
        else
        {
          if (nodsum){//enforces solid motion for the nodsum mesh nodes,
            Vm = vectorSolidMesh(nodsum,&*point,nod_vs);  //to calculate mesh velocity //ojo antes solo nod_vs
            tmp(0) = Vm(0); tmp(1) = Vm(1); if (dim == 3) tmp(2) = Vm(2);
          }
          else{//this is just an initial shoot for the mesh velocity at fluid nodes
            VecGetValues(Vec_up_0, dim, node_dofs_fluid_fs.data(), U0.data());
            VecGetValues(Vec_up_1, dim, node_dofs_fluid_fs.data(), U1.data());
            tmp = vtheta*U1 + (1.-vtheta)*U0;  //VecNorm(difff,NORM_2,&nrm);  cout << "\n" << nrm << endl;
          }
          VecSetValues(Vec_v_mid, dim, node_dofs_mesh.data(), tmp.data(), INSERT_VALUES);
        }
      } // end else if force_mesh_velocity

    } // end for point
    //VecView(Vec_v_mid,PETSC_VIEWER_STDOUT_WORLD); VecView(Vec_up_0,PETSC_VIEWER_STDOUT_WORLD); VecView(Vec_up_1,PETSC_VIEWER_STDOUT_WORLD);
  } // b.c.  //NOTE: at t=0 results Vec_v_mid=
  Assembly(Vec_v_mid); //View(Vec_v_mid,"matrizes/vmid1.m","vmidm1");
  //char buf1[50], buf2[50];
  //sprintf(buf1,"matrizes/vmid1_%d.m",time_step); sprintf(buf2,"vmidm1_%d",time_step); View(Vec_v_mid, buf1, buf2);
  if (!force_mesh_velocity)
  {
    cout << "-----Mesh solver-----" << endl;
    ierr = SNESSolve(snes_m, PETSC_NULL, Vec_v_mid);  CHKERRQ(ierr);
    cout << "---------------------" << endl;
  }
  //View(Vec_v_mid,"matrizes/vmid2.m","vmidm2");
  //sprintf(buf1,"matrizes/vmid2_%d.m",time_step); sprintf(buf2,"vmidm2_%d",time_step); View(Vec_v_mid, buf1, buf2);
  // don't let edges to be curved
  if (mesh->numNodesPerCell() > mesh->numVerticesPerCell())
  {
    int tag;
    int dofs[50];
    Vector Vm(dim);
    Vector V0(dim);
    Vector V1(dim);

    point_iterator point = mesh->pointBegin();
    point_iterator point_end = mesh->pointEnd();
    for (; point != point_end; ++point)
    {
      tag = point->getTag();

      if (mesh->isVertex(&*point))
        continue;

      if (is_in(tag, solid_tags) || is_in(tag, triple_tags) || is_in(tag, interface_tags) || is_in(tag, feature_tags))
        continue;


      const int m = point->getPosition() - mesh->numVerticesPerCell();
      Cell const* cell = mesh->getCellPtr(point->getIncidCell());
      if (dim==3)
      {
        const int edge_id = cell->getCornerId(m);
        dof_handler[DH_MESH].getVariable(VAR_M).getCornerDofs(dofs, mesh->getCornerPtr(edge_id));
      }
      else
      {
        const int edge_id = cell->getFacetId(m);
        dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(dofs, mesh->getFacetPtr(edge_id));
      }
      VecGetValues(Vec_v_mid, dim, dofs + 0*dim, V0.data());
      VecGetValues(Vec_v_mid, dim, dofs + 1*dim, V1.data());
      Vm = .5*(V0+V1);
      VecSetValues(Vec_v_mid, dim, dofs + 2*dim, Vm.data(), INSERT_VALUES);

    }
    Assembly(Vec_v_mid);
  }


  PetscFunctionReturn(0);
}

/// @brief vectorSolidMesh acts only in the calculation of the mesh velocity
/// to enforce the solid motion like advancing of the mesh for solid nodes
/// it doesn't update neither the center of mass nor the rotation angles
Vector AppCtx::vectorSolidMesh(int const K, Point const* point, int const vs)
{
  VectorXi  dofs(dim);
  Vector3d  Vm(Vector::Zero(3)), X_1(3), X_0(3), X_m1(3), X_m2(3), Vs(Vector::Zero(3));
  Matrix3d  Gamma(Matrix3d::Identity(3,3));

  getNodeDofs(&*point, DH_MESH, VAR_M, dofs.data());

  if (is_mr_qextrap || is_mr_ab){
    if (is_sfim){
      Matrix2d G1 = GammaElastMat(modes_1[K-1], 0)*GammaElastMat(modes_0[K-1], 2);
      Gamma(0,0) = G1(0,0); Gamma(1,1) = G1(1,1);
    }
    VecGetValues(Vec_x_0, dofs.size(), dofs.data(), X_0.data());
    X_1 = XG_1[K-1] + RotM(theta_1[K-1],Q_1[K-1],thetaDOF)*Gamma*RotM(theta_0[K-1],Q_0[K-1],thetaDOF).transpose()*(X_0 - XG_0[K-1]);//here it should be solve the fixed point problem
    //X_1 = XG_1[K-1] + Q_1[K-1]*Q_0[K-1].transpose()*(X_0 - XG_0[K-1]); //here it should be solve the fixed point problem
    Vm = (-X_0 + X_1)/dt;
  }
  else if (is_bdf3 && time_step > 1){
    VecGetValues(Vec_x_cur, dofs.size(), dofs.data(), X_0.data());
    VecGetValues(Vec_x_0,   dofs.size(), dofs.data(), X_m1.data());
    VecGetValues(Vec_x_aux, dofs.size(), dofs.data(), X_m2.data());
    X_1 = RotM(theta_1[K-1]-theta_0[K-1],dim)*(X_0 - XG_0[K-1]) + XG_1[K-1];
    Vm  = (11./6.*X_1 - 3.0*X_0 + 3./2.*X_m1 - 1./3.*X_m2)/dt;
  }
  else if (is_bdf2 && time_step > 0){
    if (is_bdf2_bdfe){
      VecGetValues(Vec_x_cur, dofs.size(), dofs.data(), X_0.data());
      VecGetValues(Vec_x_0,   dofs.size(), dofs.data(), X_m1.data());
      X_1 = RotM(theta_1[K-1]-theta_0[K-1],dim)*(X_0 - XG_0[K-1]) + XG_1[K-1];
      Vm  = (3./2.*X_1 - 2.0*X_0 + 1./2.*X_m1)/dt;
    }
    else if (is_bdf2_ab){
      VecGetValues(Vec_x_0, dofs.size(), dofs.data(), X_0.data());
      X_1 = RotM(theta_1[K-1]-theta_0[K-1],dim)*(X_0 - XG_0[K-1]) + XG_1[K-1];
      Vm  = (X_1 - X_0)/dt;
    }
  }
  else{  //for MR-AB and basic, and for all at time = t0
    VecGetValues(Vec_x_0, dofs.size(), dofs.data(), X_0.data());
    X_1 = RotM(theta_1[K-1]-theta_0[K-1],dim)*(X_0 - XG_0[K-1]) + XG_1[K-1];
    Vm  = (X_1 - X_0)/dt;
  }

  //if (vs && !is_sslv){
  //  VecGetValues(Vec_slipv_0, dim, dofs.data(), Vs.data());
  //  Vs = RotM(theta_1[vs-1],dim)*RotM(theta_0[vs-1],dim).transpose()*Vs;// + XG_1[nod_id+nod_is-1]; // - XG_0[nod_id+nod_is-1];
  //  VecSetValues(Vec_slipv_1, dim, dofs.data(), Vs.data(), INSERT_VALUES);
  //}

  return Vm;
}

PetscErrorCode AppCtx::updateSolidMesh()
{
  int tag, nod_id, nod_is, nod_vs, nodsum;
  VectorXi  dofs(dim), dofs_fs(LZ), dofs_U(dim);
  Vector3d  X0(Vector3d::Zero(3)), Vs(Vector3d::Zero(3));
  Vector    Zf(Vector::Zero(LZ)), Uf(dim), U(dim), VS(dim);
  vector<bool> SV(n_solids,false);  //solid visited history
  Vector3d  Xg, XG_temp, Us;
  Matrix3d  Id3(Matrix3d::Identity(3,3)), Qtmp;

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    tag = point->getTag();

    nod_id = is_in_id(tag,flusoli_tags);
    nod_is = is_in_id(tag,solidonly_tags);
    nod_vs = is_in_id(tag,slipvel_tags);
    nodsum = nod_id+nod_is+nod_vs;
    if (nodsum){
      if (!SV[nodsum-1]){ // extrapolation of mech. system's dofs
        for (int l = 0; l < LZ; l++){
          dofs_fs(l) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + l;
        }
        VecGetValues(Vec_ups_0, LZ, dofs_fs.data(), Zf.data());  //cout << Zf.transpose() << endl;
        Xg = XG_0[nodsum-1];
        Xg(0) = Xg(0)+(dt/2.0)*Zf(0); Xg(1) = Xg(1)+(dt/2.0)*Zf(1); if (dim == 3){Xg(2) = Xg(2)+(dt/2.0)*Zf(2);}
        XG_1[nodsum-1] = Xg;                               // if Zf = 0, then XG_1 = XG_0
        theta_1[nodsum-1] = theta_0[nodsum-1] + (dt/2.0)*Zf(LZ-1);  // if Zf = 0, then theta_1 = theta_0
        if (dim == 3){
          Qtmp = Id3 - (dt/2.0)*SkewMatrix(Zf.tail(3));
          invert(Qtmp,dim);
          Q_1[nodsum-1] = Qtmp*Q_0[nodsum-1];
        }
        SV[nodsum-1] = true;  //cout << XG_1[nodsum-1].transpose() <<  "   " << theta_1[nodsum-1] << endl;
      }

      getNodeDofs(&*point, DH_MESH, VAR_M, dofs.data());
      VecGetValues(Vec_x_0, dofs.size(), dofs.data(), X0.data());
      X0 = XG_1[nodsum-1] + RotM(theta_1[nodsum-1],Q_1[nodsum-1],dim)*RotM(theta_0[nodsum-1],Q_0[nodsum-1],dim).transpose()*(X0 - XG_0[nodsum-1]);// - XG_0[nodsum-1];
      VecSetValues(Vec_x_1, dofs.size(), dofs.data(), X0.data(), INSERT_VALUES);
      if (nod_vs){
        VecGetValues(Vec_slipv_0, dim, dofs.data(), Vs.data());
        Vs = RotM(theta_1[nodsum-1],Q_1[nodsum-1],dim)*RotM(theta_0[nodsum-1],Q_0[nodsum-1],dim).transpose()*Vs;// + XG_1[nod_id+nod_is-1]; // - XG_0[nod_id+nod_is-1];
        VecSetValues(Vec_slipv_1, dim, dofs.data(), Vs.data(), INSERT_VALUES);
      }
      else if (nod_id){
        for (int l = 0; l < LZ; l++){
          dofs_fs(l) = n_unknowns_u + n_unknowns_p + LZ*(nodsum-1) + l;
        }
        VecGetValues(Vec_ups_0, LZ, dofs_fs.data(), Zf.data());
        Uf = SolidVel(X0, XG_1[nod_id-1], Zf, dim);
        getNodeDofs(&*point,DH_UNKM,VAR_U,dofs_U.data());
        VecGetValues(Vec_ups_0, dim, dofs_U.data(), U.data());
        VS = U - Uf;  //cout << VS.transpose() << endl;
        VecSetValues(Vec_slipv_0, dim, dofs.data(), VS.data(), INSERT_VALUES);// before Vec_slipv_1
      }

    }
  }
  PetscFunctionReturn(0);
}

/// @brief move the implicit mesh.
///
/// x1 = Vec_x + dt*(vtheta*Vec_up_1 + (1-vtheta)*Vec_up_0)
///
/// @param[in] Vec_x_1 initial mesh
/// @param[in] Vec_up_0
/// @param[in] Vec_up_1
/// @param[in] Vec_x_new initial guess
/// @param[out] Vec_x_new
/// @warning CHANGE THE VECTOR VEC_NORMAL
PetscErrorCode AppCtx::moveMesh(Vec const& Vec_x_0, Vec const& Vec_up_0, Vec const& Vec_up_1, double const vtheta, double tt, Vec & Vec_x_new)
{
  Vector      Xnew(dim);
  Vector      X0(dim);
  Vector      tmp(dim);
  Vector      U(dim);
  Vector      U0(dim);
  Vector      U1(dim);
  VectorXi    node_dofs_mesh(dim);
  VectorXi    node_dofs_fluid(dim);
  int         tag;
  VectorXi    edge_dofs(dim);
  VectorXi    edge_nodes(3);
  //int         edge_id;
  //Cell const* cell;

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)
  {
    tag = point->getTag();

    getNodeDofs(&*point, DH_MESH, VAR_M, node_dofs_mesh.data());
    //getNodeDofs(&*point, DH_UNKS, VAR_U, node_dofs_fluid.data());
    getNodeDofs(&*point, DH_UNKM, VAR_U, node_dofs_fluid.data());
    VecGetValues(Vec_x_0,   dim, node_dofs_mesh.data(), X0.data());

    if (force_mesh_velocity)
    {
      Vector k1 = dt * v_exact(X0,tt,tag);
      Vector k2 = dt * v_exact(X0+0.5*k1,tt+0.5*dt,tag);
      Vector k3 = dt * v_exact(X0+0.5*k2,tt+0.5*dt,tag);
      Vector k4 = dt * v_exact(X0+k3,tt+dt,tag);
      tmp =  (k1 + 2.*(k2+k3) + k4)/6.; // velocity
      //if (!mesh->isVertex(&*point)) // APAGAR TEMP ERASE-ME ... este codigo é só para não entortar a malha
      //{
      //  int vtcs[3];
      //  const int m = point->getPosition() - mesh->numVerticesPerCell();
      //  Cell const* cell = mesh->getCellPtr(point->getIncidCell());
      //  if (dim==3)
      //    cell->getCornerVerticesId(m, vtcs);
      //  else
      //    cell->getFacetVerticesId(m, vtcs);
      //  if (mesh->inBoundary(mesh->getNodePtr(vtcs[0])) || mesh->inBoundary(mesh->getNodePtr(vtcs[1])))
      //    tmp = tmp / 2.;
      //  if (mesh->inBoundary(mesh->getNodePtr(vtcs[0])) && mesh->inBoundary(mesh->getNodePtr(vtcs[1])))
      //    tmp = tmp * 0.;
      //}
      tmp = X0 + tmp;
      VecSetValues(Vec_x_new, dim, node_dofs_mesh.data(), tmp.data(), INSERT_VALUES);
    }
    else
    {

      if (is_in(tag, neumann_tags) || is_in(tag, dirichlet_tags) || is_in(tag, periodic_tags))
      {
        VecSetValues(Vec_x_new, dim, node_dofs_mesh.data(), X0.data(), INSERT_VALUES);
      }
      else
      //if (is_in(tag, interface_tags) || is_in(tag, triple_tags) || is_in(tag, solid_tags) || is_in(tag,feature_tags))
      if (true)
      {
        VecGetValues(Vec_up_0,  dim, node_dofs_fluid.data(), U0.data());
        VecGetValues(Vec_up_1,  dim, node_dofs_fluid.data(), U1.data());

        tmp = X0 + dt*(  vtheta*U1 + (1.-vtheta)*U0 );
        VecSetValues(Vec_x_new, dim, node_dofs_mesh.data(), tmp.data(), INSERT_VALUES);
      }
      else // volume
      {
        VecGetValues(Vec_v_mid,   dim, node_dofs_mesh.data(), U0.data()); // get old vmesh

        //double const a = 0.001351644*X0(0)*(54.4 - X0(0));
        tmp = X0 + dt*U0 ;
        VecSetValues(Vec_x_new, dim, node_dofs_mesh.data(), tmp.data(), INSERT_VALUES);
      }
    }
  }

  Assembly(Vec_x_new);

  if (!force_mesh_velocity)
    smoothsMesh(Vec_normal, Vec_x_new);

  PetscFunctionReturn(0);
}

/// @param Vec_x the mesh
/// @param cell_id cell id
/// @return quality number in range ]-inf,1] ... 1 is the best
double AppCtx::getCellQuality(Vec const& Vec_x_, int cell_id) const
{
  MatrixXd  x_coefs(nodes_per_cell, dim);
  VectorXi  mapM_c(dim*nodes_per_cell); // mesh velocity

  dof_handler[DH_MESH].getVariable(VAR_M).getCellDofs(mapM_c.data(), mesh->getCellPtr(cell_id));

  VecGetValues(Vec_x_, mapM_c.size(), mapM_c.data(), x_coefs.data());

  if (dim==2)
  {
    double l_sqr; // l0^2 + l1^2 + l2^2

    l_sqr  = sqr(x_coefs(1,0)-x_coefs(0,0)) + sqr(x_coefs(1,1)-x_coefs(0,1));
    l_sqr += sqr(x_coefs(2,0)-x_coefs(1,0)) + sqr(x_coefs(2,1)-x_coefs(1,1));
    l_sqr += sqr(x_coefs(0,0)-x_coefs(2,0)) + sqr(x_coefs(0,1)-x_coefs(2,1));

    double const f = 6.92820323027551; // 4*sqrt(3)
    double const area = ((x_coefs(0,0)-x_coefs(2,0))*(x_coefs(1,1)-x_coefs(2,1))-(x_coefs(1,0)-x_coefs(2,0))*(x_coefs(0,1)-x_coefs(2,1)))/2;

    return f*area/l_sqr;
  }
  else // if dim==3
  {
    double l_rms_3; // pow( (l0^2 + l1^2 + l2^2 + l3^2 + l4^2 + l5^2 + l6^2)/6 , 1.5)

    l_rms_3  = sqr(x_coefs(0,0)-x_coefs(1,0))   +   sqr(x_coefs(0,1)-x_coefs(1,1))   +   sqr(x_coefs(0,2)-x_coefs(1,2));
    l_rms_3 += sqr(x_coefs(0,0)-x_coefs(2,0))   +   sqr(x_coefs(0,1)-x_coefs(2,1))   +   sqr(x_coefs(0,2)-x_coefs(2,2));
    l_rms_3 += sqr(x_coefs(0,0)-x_coefs(3,0))   +   sqr(x_coefs(0,1)-x_coefs(3,1))   +   sqr(x_coefs(0,2)-x_coefs(3,2));
    l_rms_3 += sqr(x_coefs(1,0)-x_coefs(2,0))   +   sqr(x_coefs(1,1)-x_coefs(2,1))   +   sqr(x_coefs(1,2)-x_coefs(2,2));
    l_rms_3 += sqr(x_coefs(1,0)-x_coefs(3,0))   +   sqr(x_coefs(1,1)-x_coefs(3,1))   +   sqr(x_coefs(1,2)-x_coefs(3,2));
    l_rms_3 += sqr(x_coefs(2,0)-x_coefs(3,0))   +   sqr(x_coefs(2,1)-x_coefs(3,1))   +   sqr(x_coefs(2,2)-x_coefs(3,2));
    l_rms_3 /= 6.;

    l_rms_3 = pow(l_rms_3,1.5);

    double const f = 8.48528137423857; // 6*sqrt(2)

    double x21 = x_coefs(1,0) - x_coefs(0,0);
    double x32 = x_coefs(2,0) - x_coefs(1,0);
    double x43 = x_coefs(3,0) - x_coefs(2,0);
    double y12 = x_coefs(0,1) - x_coefs(1,1);
    double y23 = x_coefs(1,1) - x_coefs(2,1);
    double y34 = x_coefs(2,1) - x_coefs(3,1);
    double z12 = x_coefs(0,2) - x_coefs(1,2);
    double z23 = x_coefs(1,2) - x_coefs(2,2);
    double z34 = x_coefs(2,2) - x_coefs(3,2);
    double volume = x21*(y23*z34 - y34*z23 ) + x32*(y34*z12 - y12*z34 ) + x43*(y12*z23 - y23*z12 );

    return f*volume/l_rms_3;
  }


}

/// @param Vec_x_ the mesh
///
double AppCtx::getCellPatchQuality(Vec const& Vec_x_, int const* cells) const
{
  double quality=99999999, aux;

  for (int i = 0; ; ++i)
  {
    if (cells[i] < 0)
      break;

    aux = getCellQuality(Vec_x_, cells[i]);
    if (aux < quality)
      quality = aux;
  }

  return quality;

}

// =====================================================================
// Luzia's methods
// =====================================================================

double AppCtx::quality_m(Mesh *mesh)
{
  int nCell=mesh->numCellsTotal();
  double Q_min=100;

  for( int i=0; i< nCell ; i++ )
  {
    Cell *cell = mesh->getCellPtr(i);
    if( !cell->isDisabled() )
    {
      double value = quality_c(cell);
      if( value < Q_min )
        Q_min=value;
    }
  }
  return Q_min;
}

double AppCtx::quality_c(Cell *cell)
{
  double Vk = MeshToolsTri::area(cell, &*mesh);
  double Pk = pow(MeshToolsTri::edgeSize(cell,0, &*mesh),2.0)+
              pow(MeshToolsTri::edgeSize(cell,1, &*mesh),2.0)+
              pow(MeshToolsTri::edgeSize(cell,2, &*mesh),2.0);
//  double hk = Pk/3;

  return 4.0*sqrt(3.0)*Vk/Pk;
  //Real min = ( hk/h_star < h_star/hk ) ? hk/h_star : h_star/hk;
  //return (20.78*Vk/(Pk*Pk)*pow(min,beta)*(2-pow(min,beta)));
}

double AppCtx::quality_e(Cell const* cell, const int j, Mesh const* mesh)
{
    Point const *a = mesh->getNodePtr(cell->getNodeId(j));
    Point const *b = mesh->getNodePtr(cell->getNodeId((j+1)%3));

    double a_coord[3], b_coord[3];
    a->getCoord(a_coord, dim);
    b->getCoord(b_coord, dim);

    double La = sizeField(a_coord), Lb = sizeField(b_coord);

    double e=MeshToolsTri::edgeSize(cell,j, mesh);
    if( La == Lb ) return e/La;

    double L = e/(La-Lb)*log(La/Lb);

    return L;
}

double AppCtx::sizeField(double *coords)
{
  double Lc = 0, dist = 1.0, L_range = 10*h_star;
  Vector3d Xg;
  for(int i = 0; i < n_solids; i++)
  {
    Xg = XG_1[i];
    double d = sqrt(pow(coords[0]-Xg(0), 2.0)+pow(coords[1]-Xg(1), 2.0));

    d=((d-RV[i](0))>0)?(d-RV[i](0)):0.0;

    dist *= fmin(1.0,d/(L_range));
  }
  //Lc = 1.2*L_min + (4*h_star-L_min)*(dist);
  Lc = L_min + (L_max-L_min)*(dist);
    //Lc=dist;

  return Lc;
}

double AppCtx::quality_f(Vector a_coord, Vector b_coord)
{
  double L = 0.0;
  double e = (a_coord-b_coord).norm();
  if (false){
    double La = sizeField_s(a_coord), Lb = sizeField_s(b_coord);
    if( La == Lb ) return e/La;
    L = e/(La-Lb)*log(La/Lb);
  }
  if (true){
    //Gauss Chebyshev (works very well, singularities)//////////////////////////////////////////////////
    double ls = 1.0, li = 0.0;//, Len = 0.0, Gen = 0.0;
    int Nroots = 100;
    for (int i = 0; i < Nroots; i++){
      double Xc = cos(pi*(2.0*(i+1)-1)/(2.0*(double)Nroots));
      double eval = (ls-li)/2.0 * Xc + (ls+li)/2.0;
      Vector XC = a_coord + eval*(b_coord - a_coord);
      L = L + 1.0/sizeField_s(XC)*sqrt(1-Xc*Xc)*e;
      //Len = Len + 1.0*sqrt(1-Xc*Xc)*e;
    }
    L = (ls-li)/2.0 * abs(L) * pi/(double)Nroots;
    //Len = (ls-li)/2.0 * abs(Len) * pi/(double)Nroots;
    //cout << "Edge length = " << e << ", estimated = " << Len <<
    //        ", for " << a_coord.transpose() << "  " << b_coord.transpose() << endl;

  }
  //cout << e << endl;
  return L;
}

double AppCtx::sizeField_s(Vector coords)
{
  double Lc = 0, dist = 1.0e10;//L_range = 10*h_star;
  Vector3d Xg, C(Vector3d::Zero(3));
  if (false){
    for(int i = 0; i < n_solids; i++)
    {
      Xg = XG_1[i];
      C(0) = coords(0); C(1) = coords(1);
      if (coords.size() == 3) {C(2) = coords(2);}
      double d = (C-Xg).norm();
      //d=((d-RV[i].maxCoeff())>0)?(d-RV[i].maxCoeff()):0.0;
      if ( d <= RV[i].maxCoeff() )
        d = 0.0;
      else if ( (d > RV[i].maxCoeff()) && (d < RV[i].maxCoeff()+L_range) )
        d = (d-RV[i].maxCoeff());
      else
        d = 1e12;
      //d = 0.0;
      dist *= fmin(1.0,d/L_range);
    }
  }
  if (true){
    int j = 0;
    for(int i = 0; i < n_solids; i++)
    {
      Xg = XG_1[i];
      C(0) = coords(0); C(1) = coords(1);
      if (coords.size() == 3) {C(2) = coords(2);}
      double d = (C-Xg).norm();
      if (d < dist){
        dist = d;
        j = i;
      }
    }
    double d = dist; int i = j;
    if ( d <= RV[i].maxCoeff() )
      d = 0.0;
    else if ( (d > RV[i].maxCoeff()) && (d < RV[i].maxCoeff()+L_range) )
      d = (d-RV[i].maxCoeff());
    else
      d = 1e12;
    dist = fmin(1.0,d/L_range);
  }
  //Lc = 1.2*L_min + (4*h_star-L_min)*(dist);
  Lc = L_min + (L_max-L_min)*(dist);
    //Lc=dist;
  return Lc;
}

PetscErrorCode AppCtx::meshAdapt_l()
{
  PetscErrorCode      ierr;

  // only for 2d ... TODO for 3d too
  if (dim != 2)
    PetscFunctionReturn(0);
  // only for linear elements
  if (mesh->numNodesPerCell() > mesh->numVerticesPerCell())
    PetscFunctionReturn(0);

  typedef tuple<int, int, int> EdgeVtcs; // get<0> = mid node, get<1> = top node, get<2> = bot node

  std::list<EdgeVtcs> adde_vtcs;

  CellElement *edge;
  Real expected_h;
  VectorXi edge_nodes(3); // 3 nodes at most
  Vector Xa(dim), Xb(dim);
  int tag_a;
  int tag_b;
  int tag_e;

  bool mesh_was_changed = false;

  double Qmesh = quality_m(&*mesh);

  if (Qmesh < Q_star){
    cout << "Entering Mesh quality test " << Qmesh << " < " << Q_star << endl;
    int nCell=mesh->numCellsTotal();

    //Collapsing//////////////////////////////////////////////////////////////////
    for(int i = 0; i < nCell ; i++)
    {
      Cell const *cell = mesh->getCellPtr(i);
      if((!cell->isDisabled())) // Testa se a célula está ativa
      {
        // Teste de qualidade, que aqui é pelo tamanho de aresta com um campo uniforme
        int minEdge = 0;
        // Determina qual a menor aresta
        double minEdgeSize = MeshToolsTri::edgeSize(cell,0,&*mesh);
        for(int j = 1; j < 3; j++)
        {
          double value = MeshToolsTri::edgeSize(cell,j,&*mesh);
          if(value < minEdgeSize)
          {
            minEdgeSize = value;
            minEdge = j;
          }
        }
        if(quality_e(cell, minEdge, &*mesh) < L_low){
          // Colapsa no centro da aresta, caso isso seja permitido
          // senão colapsa em um ponto permitido
          int eid = cell->getFacetId(minEdge);

          if (dim == 2)
            edge = mesh->getFacetPtr(eid);
          else
            edge = mesh->getCornerPtr(eid);

          if (edge==NULL || edge->isDisabled())
            continue;

          if (dim == 2)
            mesh->getFacetNodesId(eid, edge_nodes.data());
          else
            mesh->getCornerNodesId(eid, edge_nodes.data());

          tag_a = mesh->getNodePtr(edge_nodes[0])->getTag();
          tag_b = mesh->getNodePtr(edge_nodes[1])->getTag();
          tag_e = edge->getTag();

          if (is_in(tag_e, solidonly_tags) || is_in(tag_e, flusoli_tags) || is_in(tag_e, slipvel_tags))
            continue;
          if (is_in(tag_e, flusoli_tags) || is_in(tag_e, solidonly_tags))
            continue;
          if (!( tag_a==tag_b && tag_b==tag_e ))  //only edges inside the same type of domain
            continue;                                                  //(ex: inside fluid)

          Point const* pt_a = mesh->getNodePtr(edge_nodes[0]);
          Point const* pt_b = mesh->getNodePtr(edge_nodes[1]);
          //cout << edge_nodes.transpose() << endl;
          if (pt_a->isMarked() || pt_b->isMarked())
            continue;

          mesh_was_changed = true;
          //printf("COLLAPSED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
          //printf("COLLAPSING  tags = %d  %d %d #################################################\n", tag_a, tag_b, tag_e);
          int pt_id = MeshToolsTri::collapseEdge2d(edge->getIncidCell(), edge->getPosition(), 0.0, &*mesh); //0.5 para Luzia
          //printf("COLLAPSED %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", pt_id);
        }
      }
    }//end for collapse
    //Splitting//////////////////////////////////////////////////////////////////
    for( int i=0; i< nCell ; i++ )
    {
      Cell const *cell = mesh->getCellPtr(i);
      if((!cell->isDisabled())) // Testa se a célula está ativa
      {
        int maxEdge = 0;
        double maxEdgeSize = 0.0;
        for( int j=0; j<3; j++)
        {
          double value = MeshToolsTri::edgeSize(cell,j,&*mesh);
          if(value > maxEdgeSize)
          {
            maxEdge = j;
            maxEdgeSize = value;
          }
        }
        if (quality_e(cell, maxEdge, &*mesh) > L_sup){

          int eid = cell->getFacetId(maxEdge);

          if (dim == 2)
            edge = mesh->getFacetPtr(eid);
          else
            edge = mesh->getCornerPtr(eid);

          if (edge==NULL || edge->isDisabled())
            continue;

          if (dim == 2)
            mesh->getFacetNodesId(eid, edge_nodes.data());
          else
            mesh->getCornerNodesId(eid, edge_nodes.data());

          tag_a = mesh->getNodePtr(edge_nodes[0])->getTag();
          tag_b = mesh->getNodePtr(edge_nodes[1])->getTag();
          tag_e = edge->getTag();

          if (is_in(tag_e, solidonly_tags) || is_in(tag_e, flusoli_tags) || is_in(tag_e, slipvel_tags))
            continue;
          if (is_in(tag_e, flusoli_tags) || is_in(tag_e, solidonly_tags))
            continue;
          if (!( tag_a==tag_b && tag_b==tag_e ))  //only edges inside the same type of domain
            continue;                                                  //(ex: inside fluid)

          Point const* pt_a = mesh->getNodePtr(edge_nodes[0]);
          Point const* pt_b = mesh->getNodePtr(edge_nodes[1]);
          //cout << edge_nodes.transpose() << endl;
          if (pt_a->isMarked() || pt_b->isMarked())
            continue;

          pt_a->getCoord(Xa.data(),dim);
          pt_b->getCoord(Xb.data(),dim);

          expected_h = .5*(mesh_sizes[edge_nodes[0]] + mesh_sizes[edge_nodes[1]]);

          mesh_was_changed = true;
          int pt_id = MeshToolsTri::insertVertexOnEdge(edge->getIncidCell(), edge->getPosition(), 0.5, &*mesh);
          //printf("INSERTED %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", pt_id);
          adde_vtcs.push_back(make_tuple(pt_id, edge_nodes[0], edge_nodes[1]));
          mesh->getNodePtr(pt_id)->setMarkedTo(true);
          if (pt_id < (int)mesh_sizes.size())
            mesh_sizes[pt_id] = expected_h;
          else
          {
            if (pt_id > (int)mesh_sizes.size())
            {
              printf("ERROR: Something with mesh_sizes is wrong!!\n");
              throw;
            }
            mesh_sizes.push_back(expected_h);
          }
        }
      }
    }//end for splitting

  }// end if Qmesh

  if (!mesh_was_changed)
    PetscFunctionReturn(0);

  meshAliasesUpdate();
  Destroy(Mat_Jac_fs);
  Destroy(Mat_Jac_m);
  Destroy(Vec_res_fs);
  Destroy(Vec_res_m);
  Destroy(Vec_normal);
  Destroy(Vec_tangent);
  Destroy(Vec_v_mid);
  Destroy(Vec_v_1);
  SNESReset(snes_fs);
  SNESReset(snes_m);
  KSPReset(ksp_fs);
  KSPReset(ksp_m);
  PCReset(pc_fs);
  PCReset(pc_m);
  SNESLineSearchReset(linesearch);
  if (is_sslv){
    Destroy(Vec_slip_rho);
    Destroy(Vec_normal_aux);
    Destroy(Vec_rho_aux);
    Destroy(Mat_Jac_s);
    Destroy(Vec_res_s);
    SNESReset(snes_s);
    KSPReset(ksp_s);
    PCReset(pc_s);
    SNESLineSearchReset(linesearch_s);
  }
  if (time_adapt){
    Destroy(Vec_ups_time_aux);
    Destroy(Vec_x_time_aux);
  }

  DofHandler  dof_handler_tmp[2];

  // transfers variables values from old to new mesh
  {
    // First fix the u-p unknowns
    dof_handler_tmp[DH_MESH].copy(dof_handler[DH_MESH]);
    dof_handler_tmp[DH_UNKM].copy(dof_handler[DH_UNKM]);  //dof_handler_tmp[DH_UNKS].copy(dof_handler[DH_UNKS]);

    dofsUpdate();  //updates DH_ information: # variables u, z, p, mesh can change
    cout << "#z=" << n_nodes_fsi << " #u=" << n_unknowns_u << " #p=" << n_unknowns_p << " #v=" << n_dofs_v_mesh  << endl;

    Vec *petsc_vecs[] = {&Vec_ups_0,    &Vec_ups_1,    &Vec_x_0,      &Vec_x_1,      &Vec_ups_m1,   &Vec_x_aux,    &Vec_ups_m2,   &Vec_slipv_0,  &Vec_slipv_1,  &Vec_slipv_m1, &Vec_slipv_m2};
    int DH_t[]        = {DH_UNKM,       DH_UNKM,       DH_MESH,       DH_MESH,       DH_UNKM,       DH_MESH,       DH_UNKM,       DH_MESH,       DH_MESH,       DH_MESH,       DH_MESH      };
    int n_unks_t[]    = {n_unknowns_fs, n_unknowns_fs, n_dofs_v_mesh, n_dofs_v_mesh, n_unknowns_fs, n_dofs_v_mesh, n_unknowns_fs, n_dofs_v_mesh, n_dofs_v_mesh, n_dofs_v_mesh, n_dofs_v_mesh};
    int L = 4;
    if (is_slipv){L = static_cast<int>( sizeof(DH_t)/sizeof(int) );}
    else{
      if(is_bdf2){L = 5;}
      else if (is_bdf3){L = 7;}//{L = static_cast<int>( sizeof(DH_t)/sizeof(int) );}
    }
    std::vector<Real> temp;

    // NOTE: the mesh must not be changed in this loop
    for (int v = 0; v < L; ++v)
    {
      if (is_slipv){
        if ((is_mr_ab || is_basic || is_mr_qextrap) && (v == 4 || v == 5 || v == 6 || v == 9 || v == 10))
          continue;
        else if (is_bdf2 && (v == 5 || v == 6 || v == 10))
          continue;
      }

      int vsize;
      VecGetSize(*petsc_vecs[v], &vsize);  //size of Vec_ups_0, Vec_ups_1, Vec_x_0, Vec_x_1 (old mesh)
      temp.assign(vsize, 0.0);
      double *array;

      // copy
      VecGetArray(*petsc_vecs[v], &array);
      for (int i = 0; i < vsize; ++i)
        temp[i] = array[i];
      VecRestoreArray(*petsc_vecs[v], &array);
      Destroy(*petsc_vecs[v]);
      ierr = VecCreate(PETSC_COMM_WORLD, petsc_vecs[v]);                              CHKERRQ(ierr);
      ierr = VecSetSizes(*petsc_vecs[v], PETSC_DECIDE, n_unks_t[v]);                  CHKERRQ(ierr);  //dof_handler[DH_t[v]].numDofs()
      ierr = VecSetFromOptions(*petsc_vecs[v]);                                       CHKERRQ(ierr);
      if (v == 0 || v == 1 || v == 4 || v == 6){//DH_UNKM
        ierr = VecSetOption(*petsc_vecs[v],VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);
      }

      std::vector<bool>   SV(n_solids,false);  //solid visited history
      int tagP, nod_id, dofs_fs_0, dofs_fs_1, nod_vs, nodsum;

      int dofs_0[64]; // old
      int dofs_1[64]; // new
      // copy data from old mesh to new mesh
      VecGetArray(*petsc_vecs[v], &array);
      for (point_iterator point = mesh->pointBegin(), point_end = mesh->pointEnd(); point != point_end; ++point)
      {
        if (!mesh->isVertex(&*point))
          continue;

        if (!point->isMarked())
        {
          for (int k = 0; k < dof_handler_tmp[DH_t[v]].numVars(); ++k)  //ranges over # variables = 2 = u,p or 1 = mesh
          {
            dof_handler_tmp[DH_t[v]].getVariable(k).getVertexDofs(dofs_0, mesh->getPointId(&*point));
            dof_handler    [DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
            for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
              array[dofs_1[j]] = temp.at(dofs_0[j]);
          }//end for k
          if (v == 0 || v == 1 || v == 4 || v == 6){
            tagP = point->getTag();
            nod_id = is_in_id(tagP,flusoli_tags);
            nod_vs = is_in_id(tagP,slipvel_tags);
            nodsum = nod_id+nod_vs;
            if (nodsum){
              if (!SV[nodsum-1]){
                for (int l = 0; l < LZ; l++){
                  dofs_fs_0 = dof_handler_tmp[DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                              dof_handler_tmp[DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
                  dofs_fs_1 = dof_handler    [DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                              dof_handler    [DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
                  array[dofs_fs_1] = temp.at(dofs_fs_0);
                }
                SV[nodsum-1] = true;
              }
            }
          }
        }//end if !marked
      }//end for point
      // interpolate values at new points

      //if (v == 7 || v == 8 || v == 9 || v == 10)
      //  continue;

      std::list<EdgeVtcs>::iterator it     = adde_vtcs.begin();
      std::list<EdgeVtcs>::iterator it_end = adde_vtcs.end();
      for (; it != it_end; ++it)
      {
        int const pt_id        =  get<0>(*it);
        int const a_id         =  get<1>(*it);
        int const b_id         =  get<2>(*it);
        Point      * point = mesh->getNodePtr(pt_id);
        Point const* pt_a  = mesh->getNodePtr(a_id);
        Point const* pt_b  = mesh->getNodePtr(b_id);
        Point const* link[] = {pt_a, pt_b};
        int const Nlinks = sizeof(link)/sizeof(Point*);
        double const weight = 1./Nlinks;
        VectorXi  dofs_fs(LZ);
        Vector    X(dim);
        Vector    Uf(dim), Zf(LZ);

        for (int k = 0; k < dof_handler_tmp[DH_t[v]].numVars(); ++k)
        {
          dof_handler[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
          for (int j = 0; j < Nlinks; ++j)
          {
            dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, link[j]);
            for (int c = 0; c < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++c)
              array[dofs_1[c]] += weight*temp[dofs_0[c]];
          }
          if ((v == 0 || v == 1 || v == 4 || v == 6) && (k == VAR_U)){
            tagP = point->getTag();
            nod_id = is_in_id(tagP,flusoli_tags);
            nod_vs = is_in_id(tagP,slipvel_tags);
            nodsum = nod_id+nod_vs;
            if (nodsum){
              for (int l = 0; l < LZ; l++){
                dofs_fs_1 = dof_handler[DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                            dof_handler[DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
                Zf(l) = array[dofs_fs_1];
              }
              point->getCoord(X.data(),dim);
              Uf = SolidVel(X, XG_1[nodsum-1], Zf, dim);
              for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
                array[dofs_1[j]] = Uf(j);
            }
          }//end if
        }//end for k
      }//end for it
      VecRestoreArray(*petsc_vecs[v], &array);
      Assembly(*petsc_vecs[v]);
    } // end loop in vectors

    std::list<EdgeVtcs>::iterator it     = adde_vtcs.begin();
    std::list<EdgeVtcs>::iterator it_end = adde_vtcs.end();
    for (; it != it_end; ++it)
    {
      int const pt_id =  get<0>(*it);
      mesh->getNodePtr(pt_id)->setMarkedTo(false);
    }

  } // end transference

  //Vec Vec_res;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_fs);               CHKERRQ(ierr);
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
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_tangent);               CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_tangent, PETSC_DECIDE, n_dofs_v_mesh);   CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_tangent);                          CHKERRQ(ierr);

  if (is_bdf2 || is_bdf3)
  {
    //Vec Vec_x_cur; current
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_cur);              CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_x_cur, PETSC_DECIDE, n_dofs_v_mesh);  CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_x_cur);                         CHKERRQ(ierr);
  }

  if (time_adapt)
  {
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ups_time_aux);                      CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_ups_time_aux, PETSC_DECIDE, n_unknowns_fs);          CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_ups_time_aux);                                 CHKERRQ(ierr);
    ierr = VecSetOption(Vec_ups_time_aux, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_time_aux);                  CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_x_time_aux, PETSC_DECIDE, n_dofs_v_mesh);      CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_x_time_aux);                             CHKERRQ(ierr);
  }

  std::vector<int> nnz;

  {
    nnz.resize(dof_handler[DH_UNKM].numDofs()+n_solids*LZ,0);
    std::vector<SetVector<int> > tabla;
    dof_handler[DH_UNKM].getSparsityTable(tabla); // TODO: melhorar desempenho, função mt lenta
    //FEP_PRAGMA_OMP(parallel for)
     for (int i = 0; i < n_unknowns_fs - n_solids*LZ; ++i)
       nnz[i] = tabla[i].size();
  }

  //Mat Mat_Jac;
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_fs);                                      CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_fs, PETSC_DECIDE, PETSC_DECIDE, n_unknowns_fs, n_unknowns_fs);   CHKERRQ(ierr);
  ierr = MatSetFromOptions(Mat_Jac_fs);                                                 CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_fs, 0, nnz.data());                          CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_fs,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);           CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(Mat_Jac_fs, PETSC_DEFAULT, NULL);                    CHKERRQ(ierr);

  // mesh
  nnz.clear();
  int n_mesh_dofs = dof_handler[DH_MESH].numDofs();
  {
    std::vector<SetVector<int> > table;
    dof_handler[DH_MESH].getSparsityTable(table); // TODO: melhorar desempenho, função mt lenta

    nnz.resize(n_mesh_dofs, 0);

    //FEP_PRAGMA_OMP(parallel for)
    for (int i = 0; i < n_mesh_dofs; ++i)
      nnz[i] = table[i].size();

  }
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_m);                                        CHKERRQ(ierr);
  ierr = MatSetType(Mat_Jac_m,MATSEQAIJ);                                                CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_m, PETSC_DECIDE, PETSC_DECIDE, n_mesh_dofs, n_mesh_dofs);   CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_m,  0, nnz.data());                           CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);             CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_SYMMETRIC,PETSC_TRUE);                               CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fs);
  ierr = SNESSetFunction(snes_fs, Vec_res_fs, FormFunction_fs, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_fs, Mat_Jac_fs, Mat_Jac_fs, FormJacobian_fs, this);        CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(snes_fs,CheckSnesConvergence,this,PETSC_NULL);           CHKERRQ(ierr);
  ierr = SNESGetKSP(snes_fs,&ksp_fs);                                                    CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_fs,&pc_fs);                                                        CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_fs,Mat_Jac_fs,Mat_Jac_fs,SAME_NONZERO_PATTERN);             CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_fs);                                                    CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_m);
  ierr = SNESSetFunction(snes_m, Vec_res_m, FormFunction_mesh, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_m, Mat_Jac_m, Mat_Jac_m, FormJacobian_mesh, this);         CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes_m,&linesearch);                                          CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);                          CHKERRQ(ierr);
  ierr = SNESGetKSP(snes_m,&ksp_m);                                                      CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_m,&pc_m);                                                          CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_m,Mat_Jac_m,Mat_Jac_m,SAME_NONZERO_PATTERN);                CHKERRQ(ierr);
  ierr = KSPSetType(ksp_m,KSPPREONLY);                                                   CHKERRQ(ierr);
  ierr = PCSetType(pc_m,PCLU);                                                           CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_m); CHKERRQ(ierr);
  if(!nonlinear_elasticity)
  {
    ierr = SNESSetType(snes_m, SNESKSPONLY); CHKERRQ(ierr);
  }

  if (is_sslv){
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_s);                CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_res_s, PETSC_DECIDE, n_unknowns_sv);    CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_res_s);                           CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slip_rho);               CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_slip_rho, PETSC_DECIDE, n_unknowns_sv);   CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_slip_rho);                          CHKERRQ(ierr);
    ierr = VecSetOption(Vec_slip_rho, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);  CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_normal_aux);                      CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_normal_aux, PETSC_DECIDE, n_dofs_v_mesh);          CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_normal_aux);                                 CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_rho_aux);                      CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_rho_aux, PETSC_DECIDE, n_unknowns_sv);          CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_rho_aux);                                 CHKERRQ(ierr);

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

    ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_s);                                     CHKERRQ(ierr);
    ierr = MatSetType(Mat_Jac_s, MATSEQAIJ);                                            CHKERRQ(ierr);
    ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, n_pho_dofs, n_pho_dofs);  CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Mat_Jac_s,  0, nnz.data());                        CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_s,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);          CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_s,MAT_SYMMETRIC,PETSC_TRUE);                            CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD, &snes_s);                                       CHKERRQ(ierr);
    ierr = SNESSetFunction(snes_s, Vec_res_s, FormFunction_sqrm, this);                 CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes_s, Mat_Jac_s, Mat_Jac_s, FormJacobian_sqrm, this);      CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes_s,&linesearch_s);
    ierr = SNESLineSearchSetType(linesearch_s,SNESLINESEARCHBASIC);
    ierr = SNESGetKSP(snes_s,&ksp_s);                                                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_s,&pc_s);                                                       CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_s,Mat_Jac_s,Mat_Jac_s,SAME_NONZERO_PATTERN);             CHKERRQ(ierr);
    ierr = KSPSetType(ksp_s,KSPPREONLY);                                                CHKERRQ(ierr);
    ierr = PCSetType(pc_s,PCLU);                                                        CHKERRQ(ierr);
    ierr = SNESSetType(snes_s, SNESKSPONLY);                                            CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes_s);
    ierr = SNESSetType(snes_s, SNESKSPONLY);                                            CHKERRQ(ierr);
  }

  // check
  if (false)
  {
    int const n_edges_total = (dim==2) ? mesh->numFacetsTotal() : mesh->numCornersTotal();

    //printf("ENTRANDO NO LOOP eid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    for (int eid = 0; eid < n_edges_total; ++eid)
    {

      if (dim == 2)
        edge = mesh->getFacetPtr(eid);
      else
        edge = mesh->getCornerPtr(eid);

      if (edge==NULL || edge->isDisabled())
        continue;

      int tag = edge->getTag();

      if (tag != 2 && tag != 3 && tag != 7)
      {
        printf(" MERDA  MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA\n");
        printf("%d\n", tag);
        throw;
      }

      if (tag != 2 && tag != 3 && mesh->inBoundary((Facet*)edge))
      {
        printf(" MERDA  MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA\n");
        printf("%d\n", tag);
        throw;
      }

    }

  }

  checkConsistencyTri(&*mesh);

  printf("\nMesh adapted.\n\n");

  PetscFunctionReturn(0);
}

// =====================================================================
// end Luzia's methods
// =====================================================================

void AppCtx::smoothsMesh_s(Vec &Vec_normal_, Vec &Vec_x_)
{
  //int        nodeid;
  //double     *Vec_x__array;
  bool const u_has_edge_assoc_dof = shape_phi_c->numDofsAssociatedToFacet()+shape_phi_c->numDofsAssociatedToCorner() > 0;

  Vector      Xm(dim); // X mean
  Vector      X0(dim);
  Vector      Xi(dim);
  Vector      dX(dim);
  Vector      normal(dim);
  Vector      tmp(dim), tmp2(dim);
  Vector      Uf(dim), Ue(dim), Umsh(dim); // Ue := elastic velocity
  int         tag;
  int         viz_tag;
  int         iVs[128], *iVs_end;
  int         iCs[128], viCs[128];
  VectorXi    vtx_dofs_umesh(dim);  // indices de onde pegar a velocidade
  VectorXi    vtx_dofs_fluid(dim); // indices de onde pegar a velocidade
  VectorXi    edge_dofs_umesh(3*dim);
  VectorXi    edge_dofs_fluid((2+u_has_edge_assoc_dof)*dim);
  VectorXi    edge_nodes(3);
  double      error;
  double      old_quality, new_quality;
//  bool        in_boundary;
  //int        id;
//  bool        is_surface;
//  bool        is_solid;

  int const n_smooths = 10;

  /* suavização laplaciana */
  for (int smooth_it = 0; smooth_it < n_smooths; ++smooth_it)
  {
    error = 0;
    getVecNormals(&Vec_x_, Vec_normal_);

    // VERTICES
    point_iterator point = mesh->pointBegin();
    point_iterator point_end = mesh->pointEnd();
    for (; point != point_end; ++point)
    {
      tag = point->getTag();

//      is_surface = is_in(tag, interface_tags);
//      is_solid   = is_in(tag, solid_tags);

//      in_boundary =  is_surface || is_solid;

      // pula o caso em que o ponto está no contoro mas não tem boundary smoothing
//      if (!boundary_smoothing && in_boundary)
//        continue;

//      if (is_in(tag,triple_tags) || is_in(tag,feature_tags))
//        continue;

      if (is_in(tag,dirichlet_tags) || is_in(tag,flusoli_tags))
        continue;

      if (mesh->isVertex(&*point))
      {
        //if (  is_in(tag,interface_tags) || is_in(tag,triple_tags) || is_in(tag,solid_tags) ||
        //    is_in(tag,dirichlet_tags) || is_in(tag,neumann_tags)  )

        dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), &*point);
        VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), X0.data()); // old coord

        Xm = Vector::Zero(dim);
        //iVs_end = mesh->connectedVtcs(&*point, iVs);
        iVs_end = mesh->connectedVtcs(&*point, iVs, iCs, viCs);

//        if (!in_boundary)
//        {
          int N=0;
          Point const* viz_pt;
          for (int *it = iVs; it != iVs_end ; ++it)
          {
            viz_pt = mesh->getNodePtr(*it);
            viz_tag = viz_pt->getTag();
            dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), viz_pt);
            VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data());
            ++N;
            Xm += tmp;
            //if (isFixedPoint(viz_tag))
            //{
            //  Xm += 5*X0;
            //  N += 5;
            //}
          }
          Xm /= N;

          dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), &*point);
          //// compute error
          VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data()); // old coord
          error += (tmp-Xm).norm();
          old_quality = getCellPatchQuality(Vec_x_, iCs);

          VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), Xm.data(), INSERT_VALUES);

          new_quality = getCellPatchQuality(Vec_x_, iCs);

          // se a qualidade piorou, volta no que estava antes
          if (new_quality < old_quality)
          {
            VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data(), INSERT_VALUES);
          }

//        }
#if (false)
        else
        {
          int N=0;
          for (int *it = iVs; it != iVs_end ; ++it)
          {
            Point const* viz_pt = mesh->getNodePtr(*it);
            //if (viz_pt->getTag()!=tag && !is_in( viz_pt->getTag(), triple_tags ))
            viz_tag = viz_pt->getTag();
            if (viz_tag!=tag && (is_in(viz_tag,interface_tags) || is_in(viz_tag,solid_tags)) )
              continue;
            if (viz_tag!=tag && !is_in(viz_tag,triple_tags) && !isFixedPoint(viz_tag) && !is_in(viz_tag,feature_tags))
              continue;
            dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), mesh->getNodePtr(*it));
            // debug
            VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data());
            ++N;
            Xm += tmp;
            //if (isFixedPoint(viz_tag))
            //{
            //  Xm += 3*X0;
            //  N += 3;
            //}
          }

          dof_handler[DH_MESH].getVariable(VAR_M).getVertexDofs(vtx_dofs_umesh.data(), &*point);
          VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), Xi.data());

          if (dim==3)
            Xm = (N*Xi + 2*Xm)/(3*N);
            //Xm = Xm/N;
          else
            //Xm = (N*Xi + Xm)/(2*N);
            Xm = Xm/N;
          //

          dX = Xm - Xi;
          VecGetValues(Vec_normal_, dim, vtx_dofs_umesh.data(), normal.data());
          dX -= normal.dot(dX)*normal;
          Xi += dX;

          // compute error
          VecGetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data());
          error += (tmp-Xi).norm();

          //old_quality = getCellPatchQuality(Vec_x_, iCs);
          VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), Xi.data(), INSERT_VALUES);
          //new_quality = getCellPatchQuality(Vec_x_, iCs);
          //// se a qualidade piorou, volta no que estava antes
          //if (new_quality < old_quality)
          //{
          //  VecSetValues(Vec_x_, dim, vtx_dofs_umesh.data(), tmp.data(), INSERT_VALUES);
          //}
        }
#endif
      }

    } // end point

#if (false)
    // MID NODES
    point = mesh->pointBegin();
    point_end = mesh->pointEnd();
    if (u_has_edge_assoc_dof)
    for (; point != point_end; ++point)
    {
      tag = point->getTag();

      is_surface = is_in(tag, interface_tags);
      is_solid   = is_in(tag, solid_tags);

      in_boundary = mesh->inBoundary(&*point) || is_surface || is_solid;

      // pula o caso em que o ponto está no contoro mas não tem boundary smoothing
      if (!boundary_smoothing && in_boundary)
        continue;

      if (!mesh->isVertex(&*point))
      {

        const int m = point->getPosition() - mesh->numVerticesPerCell();
        Cell const* icell = mesh->getCellPtr(point->getIncidCell());
        if (dim==3)
        {
          Corner *edge = mesh->getCornerPtr(icell->getCornerId(m));
          dof_handler[DH_MESH].getVariable(VAR_M).getCornerDofs(edge_dofs_umesh.data(), &*edge);
          dof_handler[DH_UNKM].getVariable(VAR_U).getCornerDofs(edge_dofs_fluid.data(), &*edge);
          mesh->getCornerNodesId(&*edge, edge_nodes.data());
        }
        else // dim=2
        {
          Facet *edge = mesh->getFacetPtr(icell->getFacetId(m));
          dof_handler[DH_MESH].getVariable(VAR_M).getFacetDofs(edge_dofs_umesh.data(), &*edge);
          dof_handler[DH_UNKM].getVariable(VAR_U).getFacetDofs(edge_dofs_fluid.data(), &*edge);
          mesh->getFacetNodesId(&*edge, edge_nodes.data());
        }

        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data(), Xm.data());
        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data()+dim, tmp.data());
        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data()+2*dim, Xi.data());    // mid

        if (in_boundary)
        {
          Xm = (Xm+tmp+2*Xi)/4.;

          //Xm = (Xm+tmp)/2.;

          dX = Xm - Xi;
          VecGetValues(Vec_normal_, dim, edge_dofs_umesh.data()+2*dim, normal.data());
          dX -= normal.dot(dX)*normal;
          Xi += dX;
        }
        else
        {
          Xi = (Xm+tmp)/2.;
        }

        // compute error
        VecGetValues(Vec_x_, dim, edge_dofs_umesh.data()+2*dim, tmp.data());
        error += (tmp-Xi).norm();

        VecSetValues(Vec_x_, dim, edge_dofs_umesh.data()+2*dim, Xi.data(), INSERT_VALUES);
        Assembly(Vec_x_);

      }

    } // end point
#endif

    error /= mesh->numNodes();
    //if (error < 1.e-10)
    //{
    //  cout << "STOP, " << smooth_it << " iterations\n";
    //  break;
    //}


  } // end smooth
  Assembly(Vec_x_);

  getVecNormals(&Vec_x_, Vec_normal_);
}

PetscErrorCode AppCtx::meshAdapt_s()
{
  PetscErrorCode      ierr;

  // only for 2d ... TODO for 3d too
  if (dim != 2)
    PetscFunctionReturn(0);
  // only for linear elements
  if (mesh->numNodesPerCell() > mesh->numVerticesPerCell())
    PetscFunctionReturn(0);

  typedef tuple<int, int, int> EdgeVtcs; // get<0> = mid node, get<1> = top node, get<2> = bot node

  std::list<EdgeVtcs> adde_vtcs;

  CellElement *edge;
  Real h;
  Real expected_h;
  VectorXi edge_nodes(3); // 3 nodes at most
  Vector Xa(dim), Xb(dim);
  int tag_a;
  int tag_b;
  int tag_e;

  bool mesh_was_changed = false;

  double Qmesh = quality_m(&*mesh);
  //int counter = 0;
  if ((Qmesh < Q_star) /*&& (counter < 4)*/ /*true*/){

    cout << "Entering Mesh quality test: " << Qmesh << " < " << Q_star << "." << endl;

//    for (int na = 0; na < 5; na++){
    for (int is_splitting = 0; is_splitting < 2 ; ++is_splitting)
    {
      // FOR OVER EDGES' ID'S //////////////////////////////////////////////////
      int const n_edges_total = (dim==2) ? mesh->numFacetsTotal() : mesh->numCornersTotal();
      for (int eid = 0; eid < n_edges_total; ++eid)
      {
        if (dim == 2)
          edge = mesh->getFacetPtr(eid);
        else
          edge = mesh->getCornerPtr(eid);

        if (edge==NULL || edge->isDisabled())
          continue;

        if (dim == 2)
          mesh->getFacetNodesId(eid, edge_nodes.data());
        else
          mesh->getCornerNodesId(eid, edge_nodes.data());

        tag_a = mesh->getNodePtr(edge_nodes[0])->getTag();
        tag_b = mesh->getNodePtr(edge_nodes[1])->getTag();
        tag_e = edge->getTag();

        if (is_in(tag_e, solidonly_tags) || is_in(tag_e, flusoli_tags) || is_in(tag_e, slipvel_tags))
          continue;
        //if ((is_in(tag_e, flusoli_tags) || is_in(tag_e, solidonly_tags)) && (!is_splitting))
        //  continue;
        if (is_in(tag_e, dirichlet_tags))
          continue;
        //if ( !(tag_a == tag_b && tag_b == tag_e) && !is_splitting )  //only collapse edges inside the same type of domain
        //  continue;                                                //(ex: inside fluid)
        if (!is_in(tag_e,fluidonly_tags) && !is_splitting)  //only collapse edges inside fluid
          continue;

        Point const* pt_a = mesh->getNodePtr(edge_nodes[0]);
        Point const* pt_b = mesh->getNodePtr(edge_nodes[1]);
        if (pt_a->isMarked() || pt_b->isMarked())
          continue;

        pt_a->getCoord(Xa.data(),dim);
        pt_b->getCoord(Xb.data(),dim);

        h          = (Xa-Xb).norm();
        expected_h = .5*(mesh_sizes[edge_nodes[0]] + mesh_sizes[edge_nodes[1]]);


        //Collapsing (before Splitting)//////////////////////////////////////////////////
        if(!is_splitting)//&& !(time_step%2==0)) // is collapsing
        {
          bool collapsed = false, collapsed_surface = false;
          bool a_in_surface = (is_in(tag_a,slipvel_tags)||is_in(tag_a,flusoli_tags));
          bool b_in_surface = (is_in(tag_b,slipvel_tags)||is_in(tag_b,flusoli_tags));
          //cout << edge_nodes[0] << " " << edge_nodes[1] << " " << h << endl;
          //if ( is_in(tag_e, solidonly_tags) || is_in(tag_e, flusoli_tags) || is_in(tag_e, slipvel_tags) )
          //  continue;
          //if ( (h-expected_h)/expected_h < -TOLad )
          if (tag_a == tag_b && tag_b == tag_e && true){
            if (  quality_f(Xa, Xb)        < L_low  )
            {
              collapsed = true;
            }
          }
          else if ( (( a_in_surface && is_in(tag_b,fluidonly_tags)) ||
                    ( b_in_surface && is_in(tag_a,fluidonly_tags))) && false )
          {
            Cell *cell_A = mesh->getCellPtr(edge->getIncidCell());
            Cell *cell_B = mesh->getCellPtr(cell_A->getIncidCell(edge->getPosition()));
            if (quality_c(cell_A) < 0.2 || quality_c(cell_B) < 0.2){
              collapsed = true;
              collapsed_surface = true;
              //if (a_in_surface){mesh->getNodePtr(edge_nodes[0])->setBlockedTo(true);}
              //else             {mesh->getNodePtr(edge_nodes[0])->setBlockedTo(false);}
              //if (b_in_surface){mesh->getNodePtr(edge_nodes[1])->setBlockedTo(true);}
              //else             {mesh->getNodePtr(edge_nodes[1])->setBlockedTo(false);}
              //cout << "small cell at the boundary collapsed: "
              //     << "v = " << edge_nodes[0] << "," << edge_nodes[1]
              //     << ", c = " << edge->getIncidCell();

            }
          }

          if (collapsed){//////////////////////////////////////////////////
            mesh_was_changed = true;
            //int pt_id = MeshToolsTri::collapseEdge2d(edge->getIncidCell(), edge->getPosition(), 0.0, &*mesh);
            if (collapsed_surface && true){
              //int pt_id = MeshToolsTri::collapseEdge2d(edge->getIncidCell(), edge->getPosition(), 0.0, &*mesh);
              //double tcol = 0.0;
              //if (is_in(tag_a,slipvel_tags)||is_in(tag_a,flusoli_tags))
              //  tcol = 1.0;

              int pt_id = collapseEdge2d_s(edge->getIncidCell(), edge->getPosition(), 0.0, &*mesh);
              //int tag_tmp = tag_a*a_in_surface + tag_b*b_in_surface;
              //edge->setTag(10);
              //mesh->getNodePtr(pt_id)->setTag(10);
              //mesh->getNodePtr(pt_id)->setMarkedTo(true); //gives error for the Collapsing
              //mesh->getNodePtr(edge_nodes[0])->setBlockedTo(false);
              //mesh->getNodePtr(edge_nodes[1])->setBlockedTo(false);
              //cout << " id collapsed = " << pt_id;
              //if (pt_id > 0){cout << ", v tag = " << mesh->getNodePtr(pt_id)->getTag();}
              //cout << ", edge tag = " << edge->getTag() << ", c new = " << edge->getIncidCell() << endl;
              //vtk_printer_test.writeVtk();
              //vtk_printer_test.printPointTagVtk();
              //vtk_printer_test.addCellIntVtk("cell_tag", GetDataCellTag(*this));
            }
            else{
              int pt_id = MeshToolsTri::collapseEdge2d(edge->getIncidCell(), edge->getPosition(), 0.0, &*mesh);
            //printf("COLLAPSED %d !!!!!!!!!!\n", pt_id);
              //cout << " id collapsed = " << pt_id;
              //if (pt_id > 0){cout << ", v tag = " << mesh->getNodePtr(pt_id)->getTag();}
              //cout << ", edge tag = " << edge->getTag() << ", c new = " << edge->getIncidCell() << endl;
              //vtk_printer_test.writeVtk();
              //vtk_printer_test.printPointTagVtk();
            }
          }//mesh was changed//////////////////////////////////////////////////
        }
        //Splitting (after Collapsing)//////////////////////////////////////////////////
        else if (is_splitting && true)//&& (time_step%2==0))
        {
          bool splitted = false;
          if (false && is_in(tag_e, dirichlet_tags)){
            //if (false && (quality_f(Xa, Xb) > 1.2)){
            splitted = true;
            //}
          }
          //else if ( (h-expected_h)/expected_h > TOLad )
          else if (  quality_f(Xa, Xb)        > L_sup )
          {
            splitted = true;
          }
          else if ( (abs(is_in_id(tag_a,slipvel_tags)-is_in_id(tag_b,slipvel_tags)) > 0) ||
                    (abs(is_in_id(tag_a,flusoli_tags)-is_in_id(tag_b,flusoli_tags)) > 0)  )
          {
            //if (time_step == 0){
            //  splitted = true;
            //}
            if ( (fmin(is_in_id(tag_a,slipvel_tags),is_in_id(tag_b,slipvel_tags)) > 0) ||
                 (fmin(is_in_id(tag_a,flusoli_tags),is_in_id(tag_b,flusoli_tags)) > 0) ){
              splitted = true;
            }
          }

          if (splitted){//////////////////////////////////////////////////
            mesh_was_changed = true;
            int pt_id = MeshToolsTri::insertVertexOnEdge(edge->getIncidCell(), edge->getPosition(), 0.5, &*mesh);
            //printf("INSERTED %d BETWEEN %d AND %d !!!!!!!!!!", pt_id, edge_nodes[0], edge_nodes[1]); cout << " " << Xa.transpose() << " and " << Xb.transpose() << endl;
            adde_vtcs.push_back(make_tuple(pt_id, edge_nodes[0], edge_nodes[1]));
            mesh->getNodePtr(pt_id)->setMarkedTo(true);
            if (pt_id < (int)mesh_sizes.size())
              mesh_sizes[pt_id] = expected_h;
            else
            {
              if (pt_id > (int)mesh_sizes.size())
              {
                printf("ERROR: Something with mesh_sizes is wrong!!\n");
                throw;
              }
              mesh_sizes.push_back(expected_h);
            }
          }//mesh was changed//////////////////////////////////////////////////
        }//end else if//////////////////////////////////////////////////

      }
      // end n_edges_total //////////////////////////////////////////////////
    }
    //end for adaptation //////////////////////////////////////////////////
//    }
    //Qmesh = quality_m(&*mesh);  counter++;
  }
  // end if Qmesh //////////////////////////////////////////////////
  //cin.get();
  if (!mesh_was_changed){
    cout << "Mesh not adapted: Qmesh " << " > " << Q_star << " or not necessary." << endl;
    PetscFunctionReturn(0);
  }

  //if (true) {CheckInvertedElements();}

  meshAliasesUpdate();

  Destroy(Mat_Jac_fs);
  Destroy(Mat_Jac_m);

  // No Interpolated Vectors in Mesh Adaptation: Destroy them //////////////////////////////////////////////////
  Destroy(Vec_res_fs);
  Destroy(Vec_res_m);
  //Destroy(Vec_v_mid);
  Destroy(Vec_v_1);
  Destroy(Vec_normal);
  Destroy(Vec_tangent);
  if (is_bdf2 || is_bdf3){Destroy(Vec_x_cur);}
  if (time_adapt){
    Destroy(Vec_ups_time_aux);
    Destroy(Vec_x_time_aux);
  }

  // pestsc contexts //////////////////////////////////////////////////
  SNESReset(snes_fs);
  SNESReset(snes_m);
  KSPReset(ksp_fs);
  KSPReset(ksp_m);
  PCReset(pc_fs);
  PCReset(pc_m);
  SNESLineSearchReset(linesearch);

  // Swimmer //////////////////////////////////////////////////
  if (is_sslv){
    Destroy(Vec_res_s);
    Destroy(Vec_slip_rho);
    Destroy(Vec_normal_aux);
    Destroy(Vec_rho_aux);
    Destroy(Mat_Jac_s);
    SNESReset(snes_s);
    KSPReset(ksp_s);
    PCReset(pc_s);
    SNESLineSearchReset(linesearch_s);
  }

  // Dissipative Force //////////////////////////////////////////////////
  if (is_sfip){
    Destroy(Vec_res_Fdis);
    Destroy(Vec_fdis_0);
    Destroy(Vec_ftau_0);
    Destroy(Mat_Jac_fd);
    SNESReset(snes_fd);
    KSPReset(ksp_fd);
    PCReset(pc_fd);
    SNESLineSearchReset(linesearch_fd);
  }


  // Interpolated Vectors in Mesh Adaptation: Modify them //////////////////////////////////////////////////
  DofHandler  dof_handler_tmp[2];

  {// transfering variables values from old to new mesh //////////////////////////////////////////////////

    // First fix the u-p unknowns
    dof_handler_tmp[DH_MESH].copy(dof_handler[DH_MESH]);
    dof_handler_tmp[DH_UNKM].copy(dof_handler[DH_UNKM]);  //dof_handler_tmp[DH_UNKS].copy(dof_handler[DH_UNKS]);
    dofsUpdate();  //updates DH_ information: # variables u, z, p, mesh can change
    cout << " #u=" << n_unknowns_u << " #p=" << n_unknowns_p << " #z=" << n_unknowns_z << " #l=" << n_links << " #m=" << n_modes
         << " #v=" << n_dofs_v_mesh  << " #n_fsi=" << n_nodes_fsi << " #n_facets_fsi=" << n_facets_fsi << " #n_nodes_surface=" << n_nodes_sv+n_nodes_fs << endl;

    Vec *petsc_vecs[] = {&Vec_ups_0,    &Vec_ups_1,    &Vec_x_0,      &Vec_x_1,      &Vec_v_mid,    &Vec_ups_m1,   &Vec_x_aux,    &Vec_ups_m2,   &Vec_slipv_0,  &Vec_slipv_1,  &Vec_slipv_m1, &Vec_slipv_m2};
    int DH_t[]        = {DH_UNKM,       DH_UNKM,       DH_MESH,       DH_MESH,       DH_MESH,       DH_UNKM,       DH_MESH,       DH_UNKM,       DH_MESH,       DH_MESH,       DH_MESH,       DH_MESH      };
    int n_unks_t[]    = {n_unknowns_fs, n_unknowns_fs, n_dofs_v_mesh, n_dofs_v_mesh, n_dofs_v_mesh, n_unknowns_fs, n_dofs_v_mesh, n_unknowns_fs, n_dofs_v_mesh, n_dofs_v_mesh, n_dofs_v_mesh, n_dofs_v_mesh};
    int L = 5;
    if (is_sfip){L = static_cast<int>( sizeof(DH_t)/sizeof(int) );}
    else{
      if(is_bdf2){L = 6;}
      else if (is_bdf3){L = 8;}//{L = static_cast<int>( sizeof(DH_t)/sizeof(int) );}
    }
    std::vector<Real> temp;

    // NOTE: the mesh must not be changed in this loop
    for (int v = 0; v < L; ++v)  //for each vector to interpolate
    {
      if (is_sfip){
        if ((is_mr_ab || is_basic || is_mr_qextrap) && (/*v == 5 ||*/ /*v == 6 ||*/ v == 7 || v == 10 || v == 11))
          continue;
        else if (is_bdf2 && (v == 5 || v == 7 || v == 11))
          continue;
      }

      //copy old values to array and then to temp//////////////////////////////////////////////////
      int vsize;
      VecGetSize(*petsc_vecs[v], &vsize);  //size of Vec_ups_0, Vec_ups_1, Vec_x_0, Vec_x_1 (old mesh)
      temp.assign(vsize, 0.0);

      double *array;
      VecGetArray(*petsc_vecs[v], &array);

      for (int i = 0; i < vsize; ++i)
        temp[i] = array[i];
      VecRestoreArray(*petsc_vecs[v], &array);
      Destroy(*petsc_vecs[v]);
      ierr = VecCreate(PETSC_COMM_WORLD, petsc_vecs[v]);                              CHKERRQ(ierr);
      ierr = VecSetSizes(*petsc_vecs[v], PETSC_DECIDE, n_unks_t[v]);                  CHKERRQ(ierr);  //dof_handler[DH_t[v]].numDofs()
      ierr = VecSetFromOptions(*petsc_vecs[v]);                                       CHKERRQ(ierr);
      if (v == 0 || v == 1 || v == 5 || v == 7){//DH_UNKM
        ierr = VecSetOption(*petsc_vecs[v],VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////

      //copy data from old mesh to new mesh//////////////////////////////////////////////////
      std::vector<bool>   SV(n_solids,false);  //solid visited history
      int tagP, nod_id, dofs_fs_0, dofs_fs_1, nod_vs, nodsum;

      int dofs_0[64]; // old
      int dofs_1[64]; // new
      VecGetArray(*petsc_vecs[v], &array);
      for (point_iterator point = mesh->pointBegin(), point_end = mesh->pointEnd(); point != point_end; ++point)
      {
        if (!mesh->isVertex(&*point))
          continue;

        if (!point->isMarked())
        {
          for (int k = 0; k < dof_handler_tmp[DH_t[v]].numVars(); ++k)  //ranges over # variables = 2 = u,p or 1 = mesh
          {
            dof_handler_tmp[DH_t[v]].getVariable(k).getVertexDofs(dofs_0, mesh->getPointId(&*point));
            dof_handler    [DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
            for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
              array[dofs_1[j]] = temp.at(dofs_0[j]);
          }//end for k
/*          if (v == 0 || v == 1 || v == 5 || v == 7){
            tagP = point->getTag();
            nod_id = is_in_id(tagP,flusoli_tags);
            nod_vs = is_in_id(tagP,slipvel_tags);
            nodsum = nod_id+nod_vs;
            if (nodsum){
              if (!SV[nodsum-1]){
                for (int l = 0; l < LZ; l++){
                  dofs_fs_0 = dof_handler_tmp[DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                              dof_handler_tmp[DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
                  dofs_fs_1 = dof_handler    [DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                              dof_handler    [DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
                  array[dofs_fs_1] = temp.at(dofs_fs_0);
                }
                SV[nodsum-1] = true;
              }
            }
          }*/
        }//end if !marked
      }//end for point

      if (is_sfip){//transport soild's DOFs information to the new vectors////////////////////
        if (v == 0 || v == 1 || v == 5 || v == 7){
          for (int K = 1; K <= n_solids ; K++){
            nodsum = K;
            for (int l = 0; l < LZ; l++){
              dofs_fs_0 = dof_handler_tmp[DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                          dof_handler_tmp[DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
              dofs_fs_1 = dof_handler    [DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                          dof_handler    [DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
              array[dofs_fs_1] = temp.at(dofs_fs_0);
            }
          }//end K

          if (is_sflp){//transport links////////////////////
            for (int nl = 0; nl < n_links; nl++){
              dofs_fs_0 = dof_handler_tmp[DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                          dof_handler_tmp[DH_t[v]].getVariable(VAR_P).numPositiveDofs() + n_unknowns_z + nl;
              dofs_fs_1 = dof_handler    [DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                          dof_handler    [DH_t[v]].getVariable(VAR_P).numPositiveDofs() + n_unknowns_z + nl;
              array[dofs_fs_1] = temp.at(dofs_fs_0);
            }
          }//end sflp

        }//end v
      }//end sfip

      //no interpolation for slip velocity vector////////////////////
      if (v == 8 || v == 9 || v == 10 || v == 11){continue;}
      //////////////////////////////////////////////////

      //interpolate values at new points//////////////////////////////////////////////////
      std::list<EdgeVtcs>::iterator it     = adde_vtcs.begin();
      std::list<EdgeVtcs>::iterator it_end = adde_vtcs.end();
      for (; it != it_end; ++it)
      {
        int const pt_id        =  get<0>(*it);
        int const a_id         =  get<1>(*it);
        int const b_id         =  get<2>(*it);
        Point      * point = mesh->getNodePtr(pt_id);
        Point const* pt_a  = mesh->getNodePtr(a_id);
        Point const* pt_b  = mesh->getNodePtr(b_id);
        Point const* link[] = {pt_a, pt_b};
        int const Nlinks = sizeof(link)/sizeof(Point*);
        double const weight = 1./Nlinks;
        VectorXi  dofs_fs(LZ);
        Vector    X(dim);
        Vector    Uf(dim), Zf(LZ);

        for (int k = 0; k < dof_handler_tmp[DH_t[v]].numVars(); ++k)
        {
          dof_handler[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_1, &*point);
          for (int j = 0; j < Nlinks; ++j)
          {
            dof_handler_tmp[DH_t[v]].getVariable(k).getVertexAssociatedDofs(dofs_0, link[j]);
            for (int c = 0; c < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++c)
              array[dofs_1[c]] += weight*temp[dofs_0[c]];
          }
          if ((v == 0 || v == 1 || v == 5 || v == 7) && (k == VAR_U)){//if the new node is in the solid////////////
            tagP = point->getTag();
            nod_id = is_in_id(tagP,flusoli_tags);
            nod_vs = is_in_id(tagP,slipvel_tags);
            nodsum = nod_id+nod_vs;
            if (nodsum){
              for (int l = 0; l < LZ; l++){
                dofs_fs_1 = dof_handler[DH_t[v]].getVariable(VAR_U).numPositiveDofs() +
                            dof_handler[DH_t[v]].getVariable(VAR_P).numPositiveDofs() + LZ*(nodsum-1) + l;
                Zf(l) = array[dofs_fs_1];
              }
              point->getCoord(X.data(),dim);
              Uf = SolidVel(X, XG_1[nodsum-1], Zf, dim);
              for (int j = 0; j < dof_handler_tmp[DH_t[v]].getVariable(k).numDofsPerVertex(); ++j)
                array[dofs_1[j]] = Uf(j);
            }
          }//end if
        }//end for k
      }//end for it
      VecRestoreArray(*petsc_vecs[v], &array);
      Assembly(*petsc_vecs[v]);
    } // end loop in vectors

    std::list<EdgeVtcs>::iterator it     = adde_vtcs.begin();
    std::list<EdgeVtcs>::iterator it_end = adde_vtcs.end();
    for (; it != it_end; ++it)
    {
      int const pt_id =  get<0>(*it);
      mesh->getNodePtr(pt_id)->setMarkedTo(false);
    }

  } // end transference

  allocPetscObjsVecNoInt();

/*  //Vec Vec_res;
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_fs);               CHKERRQ(ierr);
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
  ierr = VecCreate(PETSC_COMM_WORLD, &Vec_tangent);               CHKERRQ(ierr);
  ierr = VecSetSizes(Vec_tangent, PETSC_DECIDE, n_dofs_v_mesh);   CHKERRQ(ierr);
  ierr = VecSetFromOptions(Vec_tangent);                          CHKERRQ(ierr);

  if (is_bdf2 || is_bdf3)
  {
    //Vec Vec_x_cur; current
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_cur);              CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_x_cur, PETSC_DECIDE, n_dofs_v_mesh);  CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_x_cur);                         CHKERRQ(ierr);
  }

  if (time_adapt)
  {
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_ups_time_aux);                      CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_ups_time_aux, PETSC_DECIDE, n_unknowns_fs);          CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_ups_time_aux);                                 CHKERRQ(ierr);
    ierr = VecSetOption(Vec_ups_time_aux, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_x_time_aux);                  CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_x_time_aux, PETSC_DECIDE, n_dofs_v_mesh);      CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_x_time_aux);                             CHKERRQ(ierr);
  }*/

//  std::vector<int> nnz;

/*  // mesh
  nnz.clear();
  int n_mesh_dofs = dof_handler[DH_MESH].numDofs();
  {
    std::vector<SetVector<int> > table;
    dof_handler[DH_MESH].getSparsityTable(table); // TODO: melhorar desempenho, função mt lenta

    nnz.resize(n_mesh_dofs, 0);

    //FEP_PRAGMA_OMP(parallel for)
    for (int i = 0; i < n_mesh_dofs; ++i)
      nnz[i] = table[i].size();

  }
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_m);                                        CHKERRQ(ierr);
  ierr = MatSetType(Mat_Jac_m,MATSEQAIJ);                                                CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_m, PETSC_DECIDE, PETSC_DECIDE, n_mesh_dofs, n_mesh_dofs);   CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_m,  0, nnz.data());                           CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);             CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_m,MAT_SYMMETRIC,PETSC_TRUE);                               CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_m);
  ierr = SNESSetFunction(snes_m, Vec_res_m, FormFunction_mesh, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_m, Mat_Jac_m, Mat_Jac_m, FormJacobian_mesh, this);         CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes_m,&linesearch);                                          CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);                          CHKERRQ(ierr);
  ierr = SNESGetKSP(snes_m,&ksp_m);                                                      CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_m,&pc_m);                                                          CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_m,Mat_Jac_m,Mat_Jac_m,SAME_NONZERO_PATTERN);                CHKERRQ(ierr);
  ierr = KSPSetType(ksp_m,KSPGMRES);                                                     CHKERRQ(ierr); //gmres iterative method for mesh
  //ierr = KSPSetType(ksp_m,KSPPREONLY);                                                   CHKERRQ(ierr);
  ierr = PCSetType(pc_m,PCLU);                                                           CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_m); CHKERRQ(ierr);
  if(false && !nonlinear_elasticity)
  {
    ierr = SNESSetType(snes_m, SNESKSPONLY); CHKERRQ(ierr);
  }
*/

//  {
//    nnz.resize(n_unknowns_fs /*dof_handler[DH_UNKM].numDofs()+n_solids*LZ*/,0);
//    std::vector<SetVector<int> > tabla;
//    dof_handler[DH_UNKM].getSparsityTable(tabla); // TODO: melhorar desempenho, função mt lenta
//    //FEP_PRAGMA_OMP(parallel for)
//      for (int i = 0; i < n_unknowns_u + n_unknowns_p /*n_unknowns_fs - n_solids*LZ*/; ++i)
//        nnz[i] = tabla[i].size();
//  }
/*
  //Mat Mat_Jac;
  ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_fs);                                      CHKERRQ(ierr);
  ierr = MatSetSizes(Mat_Jac_fs, PETSC_DECIDE, PETSC_DECIDE, n_unknowns_fs, n_unknowns_fs);   CHKERRQ(ierr);
  ierr = MatSetFromOptions(Mat_Jac_fs);                                                 CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Mat_Jac_fs, 0, nnz.data());                          CHKERRQ(ierr);
  ierr = MatSetOption(Mat_Jac_fs,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);           CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fs);
  ierr = SNESSetFunction(snes_fs, Vec_res_fs, FormFunction_fs, this);                    CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_fs, Mat_Jac_fs, Mat_Jac_fs, FormJacobian_fs, this);        CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(snes_fs,CheckSnesConvergence,this,PETSC_NULL);           CHKERRQ(ierr);
  ierr = SNESGetKSP(snes_fs,&ksp_fs);                                                    CHKERRQ(ierr);
  ierr = KSPGetPC(ksp_fs,&pc_fs);                                                        CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_fs,Mat_Jac_fs,Mat_Jac_fs,SAME_NONZERO_PATTERN);             CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_fs);                                                    CHKERRQ(ierr);
*/
  allocPetscObjsMesh();
  allocPetscObjsFluid();

  if (is_sslv){
    allocPetscObjsSwimmer();
/*    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_s);                CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_res_s, PETSC_DECIDE, n_unknowns_sv);    CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_res_s);                           CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_slip_rho);               CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_slip_rho, PETSC_DECIDE, n_unknowns_sv);   CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_slip_rho);                          CHKERRQ(ierr);
    ierr = VecSetOption(Vec_slip_rho, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);  CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_normal_aux);                      CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_normal_aux, PETSC_DECIDE, n_dofs_v_mesh);          CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_normal_aux);                                 CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_rho_aux);                      CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_rho_aux, PETSC_DECIDE, n_unknowns_sv);          CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_rho_aux);                                 CHKERRQ(ierr);

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

    ierr = MatCreate(PETSC_COMM_WORLD, &Mat_Jac_s);                                     CHKERRQ(ierr);
    ierr = MatSetType(Mat_Jac_s, MATSEQAIJ);                                            CHKERRQ(ierr);
    ierr = MatSetSizes(Mat_Jac_s, PETSC_DECIDE, PETSC_DECIDE, n_pho_dofs, n_pho_dofs);  CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Mat_Jac_s,  0, nnz.data());                        CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_s,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);          CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_s,MAT_SYMMETRIC,PETSC_TRUE);                            CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD, &snes_s);                                       CHKERRQ(ierr);
    ierr = SNESSetFunction(snes_s, Vec_res_s, FormFunction_sqrm, this);                 CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes_s, Mat_Jac_s, Mat_Jac_s, FormJacobian_sqrm, this);      CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes_s,&linesearch_s);
    ierr = SNESLineSearchSetType(linesearch_s,SNESLINESEARCHBASIC);
    ierr = SNESGetKSP(snes_s,&ksp_s);                                                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_s,&pc_s);                                                       CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_s,Mat_Jac_s,Mat_Jac_s,SAME_NONZERO_PATTERN);             CHKERRQ(ierr);
    ierr = KSPSetType(ksp_s,KSPPREONLY);                                                CHKERRQ(ierr);
    ierr = PCSetType(pc_s,PCLU);                                                        CHKERRQ(ierr);
    ierr = SNESSetType(snes_s, SNESKSPONLY);                                            CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes_s);
    ierr = SNESSetType(snes_s, SNESKSPONLY);                                            CHKERRQ(ierr);*/
  }

  if (is_sfip){
    allocPetscObjsDissForce();
/*
    //  ------------------------------------------------------------------
    //  ----------------------- Dissipative Force ------------------------
    //  ------------------------------------------------------------------

    //Vec Vec_fdis_0;
    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_fdis_0);               CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_fdis_0, PETSC_DECIDE, n_dofs_v_mesh);   CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_fdis_0);                          CHKERRQ(ierr);
    ierr = VecSetOption(Vec_fdis_0, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);  CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &Vec_res_Fdis);                CHKERRQ(ierr);
    ierr = VecSetSizes(Vec_res_Fdis, PETSC_DECIDE, n_dofs_v_mesh);    CHKERRQ(ierr);
    ierr = VecSetFromOptions(Vec_res_Fdis);                           CHKERRQ(ierr);

    nnz.clear();
    int n_fd_dofs = dof_handler[DH_MESH].numDofs();
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
    ierr = MatSetOption(Mat_Jac_fd,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);             CHKERRQ(ierr);
    ierr = MatSetOption(Mat_Jac_fd,MAT_SYMMETRIC,PETSC_TRUE);                               CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD, &snes_fd);                                          CHKERRQ(ierr);
    ierr = SNESSetFunction(snes_fd,Vec_res_Fdis,FormFunction_fd,this);                    CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes_fd,Mat_Jac_fd,Mat_Jac_fd,FormJacobian_fd,this);         CHKERRQ(ierr);
    ierr = SNESGetLineSearch(snes_fd,&linesearch_fd);
    ierr = SNESLineSearchSetType(linesearch_fd,SNESLINESEARCHBASIC);
    ierr = SNESGetKSP(snes_fd,&ksp_fd);                                                   CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_fd,&pc_fd);                                                       CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_fd,Mat_Jac_fd,Mat_Jac_fd,SAME_NONZERO_PATTERN);             CHKERRQ(ierr);
    ierr = KSPSetType(ksp_fd,KSPPREONLY);                                                CHKERRQ(ierr);
    ierr = PCSetType(pc_fd,PCLU);                                                        CHKERRQ(ierr);
    //ierr = SNESSetFromOptions(snes_fd);                                                  CHKERRQ(ierr);
    ierr = SNESSetType(snes_fd, SNESKSPONLY);                                            CHKERRQ(ierr);
    //cout << endl; ierr = SNESView(snes_s, PETSC_VIEWER_STDOUT_WORLD);
*/
  }


  // check
  if (false)
  {
    int const n_edges_total = (dim==2) ? mesh->numFacetsTotal() : mesh->numCornersTotal();

    //printf("ENTRANDO NO LOOP eid!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    for (int eid = 0; eid < n_edges_total; ++eid)
    {

      if (dim == 2)
        edge = mesh->getFacetPtr(eid);
      else
        edge = mesh->getCornerPtr(eid);

      if (edge==NULL || edge->isDisabled())
        continue;

      int tag = edge->getTag();

      if (tag != 2 && tag != 3 && tag != 7)
      {
        printf(" MERDA  MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA\n");
        printf("%d\n", tag);
        throw;
      }

      if (tag != 2 && tag != 3 && mesh->inBoundary((Facet*)edge))
      {
        printf(" MERDA  MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA MERDA\n");
        printf("%d\n", tag);
        throw;
      }

    }

  }

  checkConsistencyTri(&*mesh);

  printf("Mesh adapted.\n");

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::meshFlipping_s()
{
  // MESH FLIPPING
  if(time_step%1 == 0)
  {
    if (dim==2)
    {
      Facet *f;
      Cell* ca, *cb;
      // Delaunay
      for (int i = 0; i < n_facets_total; ++i)
      {
        f = mesh->getFacetPtr(i);
        if (f->isDisabled())
          continue;
        if (!MeshToolsTri::inCircle2d(f, &*mesh))
        {
          ca = mesh->getCellPtr(f->getIncidCell());
          cb = mesh->getCellPtr(ca->getIncidCell(f->getPosition()));

          if (cb)
            if (ca->getTag() == cb->getTag())
              MeshToolsTri::flipEdge(f, &*mesh, true);
        }
      }
    } // end dim==2
  } // end ts%1
  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::getMeshSizes()
{

  mesh_sizes.reserve(n_nodes_total + 1*n_nodes_total/10);  //modify the capacity of mesh_sizes
  mesh_sizes.assign(n_nodes_total, 0.0);  //put zeros in the mesh_size vector (n_nodes_total components)

  // stores how much edges were connected until now at each vertex
  std::vector<int> counter(n_nodes_total, 0);

  VectorXi edge_nodes(3);
  Vector Xa(dim), Xb(dim);
  Real h;

  const int n_edges_total = (dim==2) ? mesh->numFacetsTotal() : mesh->numCornersTotal();
  CellElement const* edge(NULL);

  //FEP_PRAGMA_OMP(for nowait)
  for (int a = 0; a < n_edges_total; ++a)  //total edges ordered by element ID in .geo
  {
    if (dim == 2)
      edge = mesh->getFacetPtr(a);
    else
      edge = mesh->getCornerPtr(a);

    if (edge==NULL || edge->isDisabled())
      continue;

    if (dim == 2)
      mesh->getFacetNodesId(a, edge_nodes.data());
    else
      mesh->getCornerNodesId(a, edge_nodes.data());

    mesh->getNodePtr(edge_nodes[0])->getCoord(Xa.data(),dim);  //coords node1 of current edge
    mesh->getNodePtr(edge_nodes[1])->getCoord(Xb.data(),dim);  //coords node2 of current edge
    h = (Xa-Xb).norm();  //cout << h << "     " << edge_nodes.transpose() << "     ";

    mesh_sizes[edge_nodes[0]] = (mesh_sizes[edge_nodes[0]]*counter[edge_nodes[0]] + h)/(counter[edge_nodes[0]] + 1.0);  //cout << mesh_sizes[edge_nodes[0]] << "  ";
    mesh_sizes[edge_nodes[1]] = (mesh_sizes[edge_nodes[1]]*counter[edge_nodes[1]] + h)/(counter[edge_nodes[1]] + 1.0);  //cout << mesh_sizes[edge_nodes[1]] << endl;

    ++counter[edge_nodes[0]];
    ++counter[edge_nodes[1]];
  } // end n_edges_total //cout << mesh_sizes[2] << " " << mesh_sizes[3] << endl;

  PetscFunctionReturn(0);
}

PetscErrorCode AppCtx::orthogTest(Vec const& Vec_0, Vec const& Vec_1)
{
  int         tag;
  VectorXi    node_dofs_mesh(dim);
  Vector      X0(dim);
  Vector      X1(dim);

  point_iterator point = mesh->pointBegin();
  point_iterator point_end = mesh->pointEnd();
  for (; point != point_end; ++point)  //to calculate Vec_v_mid at each point (initial guess)
  {
    tag = point->getTag();
    if (is_in(tag,slipvel_tags)){
      getNodeDofs(&*point, DH_MESH, VAR_M, node_dofs_mesh.data());
      VecGetValues(Vec_0,dim,node_dofs_mesh.data(),X0.data());
      VecGetValues(Vec_1,dim,node_dofs_mesh.data(),X1.data());
      cout << mesh->getPointId(&*point) << " " << X0.dot(X1) << "; ";
    }
  }
  PetscFunctionReturn(0);
}

int AppCtx::collapseEdge2d_s(int cell_A_id, int face_Am_id, Real t, Mesh *mesh)
{
    int const  sdim = mesh->spaceDim();

    // ============================== Ids das celulas opostas à aresta =================================================
    Cell const *cell_A = mesh->getCellPtr(cell_A_id);
    int const cell_B_id = cell_A->getIncidCell(face_Am_id);
    Cell *cell_B = NULL;

    // ================================ Teste se a aresta é de bordo ===================================================
    const bool edge_in_boundary = cell_B_id < 0;

    // ================================= Ainda não tratei este caso ====================================================
    //const bool highm_order = mesh->numNodesPerCell() > mesh->numVerticesPerCell();

    // =============================== Ids locais das arestas das celulas ==============================================
    int face_Abr_id = (face_Am_id+1)%3,
        face_Atr_id = (face_Am_id+2)%3,
        face_Bbl_id = -1,
        face_Btl_id = -1,
        face_Bm_id   = -1;

    if ( !edge_in_boundary )
    { // Trata o caso onde a aresta não é de bordo
        cell_B       = mesh->getCellPtr(cell_B_id);  // Ponteiro para a celula B
        face_Bm_id   = cell_A->getIncidCellPos(face_Am_id);
        face_Btl_id  = (face_Bm_id+1)%3;
        face_Bbl_id  = (face_Bm_id+2)%3;
    }

    // =============================== Ids dos vertices ================================================================
    int const vtx_t_id = cell_A->getNodeId(face_Am_id);
    int const vtx_b_id = cell_A->getNodeId(face_Abr_id);
    int const vtx_r_id = cell_A->getNodeId(face_Atr_id);
    int       vtx_l_id = -1;
    if ( !edge_in_boundary )
    { // Trata o caso onde a aresta não é de bordo
        vtx_l_id = cell_B->getNodeId(face_Bbl_id);
    }
    int tag_tmp = mesh->getNodePtr(vtx_t_id)->getTag();
    if (is_in(tag_tmp,slipvel_tags)||is_in(tag_tmp,flusoli_tags)) {/*cin.get();*/ return -1;}

    // ================================ Id global das arestas ==========================================================
    int const edge_m_id  = cell_A->getFacetId(face_Am_id); // = old edge
    int       edge_br_id = cell_A->getFacetId(face_Abr_id); // Aresta que segue o vertice vtx_b
    int       edge_tr_id = cell_A->getFacetId(face_Atr_id); // Aresta que segue o vtx_l
    int       edge_bl_id = -1; // Aresta que segue o vertice vtx_r
    int       edge_tl_id = -1; // Aresta entre o vertices vtx_b e vtx_r

    if ( !edge_in_boundary )
    { // Trata o caso onde a aresta não é de bordo
        edge_bl_id = cell_B->getFacetId(face_Bbl_id);
        edge_tl_id = cell_B->getFacetId(face_Btl_id);
    }
/*
    VectorXi edge_nodes(3); cout << endl;// 3 nodes at most
    mesh->getFacetNodesId(edge_m_id, edge_nodes.data());
    cout << "facet id " << edge_m_id << " nodes " << edge_nodes[0] << "," << edge_nodes[1] << endl;
    mesh->getFacetNodesId(edge_br_id, edge_nodes.data());
    cout << "facet id " << edge_br_id << " nodes " << edge_nodes[0] << "," << edge_nodes[1] << endl;
    mesh->getFacetNodesId(edge_tr_id, edge_nodes.data());
    cout << "facet id " << edge_tr_id << " nodes " << edge_nodes[0] << "," << edge_nodes[1] << endl;
    mesh->getFacetNodesId(edge_bl_id, edge_nodes.data());
    cout << "facet id " << edge_bl_id << " nodes " << edge_nodes[0] << "," << edge_nodes[1] << endl;
    mesh->getFacetNodesId(edge_tl_id, edge_nodes.data());
    cout << "facet id " << edge_tl_id << " nodes " << edge_nodes[0] << "," << edge_nodes[1] << endl;
*/
    // ========================== Guada o Id global das celulas vizinhas ===============================================
    int const cell_AT_id = cell_A->getIncidCell(face_Atr_id),
              cell_AB_id = cell_A->getIncidCell(face_Abr_id);
    int       cell_BT_id = -1,
              cell_BB_id = -1;
    if ( !edge_in_boundary )
    { // Trata o caso onde a aresta não é de bordo
        cell_BT_id = cell_B->getIncidCell(face_Btl_id);
        cell_BB_id = cell_B->getIncidCell(face_Bbl_id);
    }

    // =========================== Guarda o Id local das faces nas celulas vizinhas ====================================
    int const face_ATtr_id = cell_A->getIncidCellPos(face_Atr_id),
              face_ABbr_id = cell_A->getIncidCellPos(face_Abr_id);
    int       face_BTtl_id = -1,
              face_BBbl_id = -1;
    if ( !edge_in_boundary )
    { // Trata o caso onde a aresta não é de bordo
        face_BTtl_id = cell_B->getIncidCellPos(face_Btl_id);
        face_BBbl_id = cell_B->getIncidCellPos(face_Bbl_id);
    }

    // TO DO
    // ================================ Trata as componentes conexa e de bordo =========================================
    // int bdr=mesh->getFacetPtr(edge_tr_id)->getBoundaryComponentId();

    // ================================ Mover os vertices vtx_t e vtx_b para a posição de colapso ======================
    Point *vtx_t = mesh->getNodePtr(vtx_t_id),
          *vtx_b = mesh->getNodePtr(vtx_b_id),
          *vtx_r = mesh->getNodePtr(vtx_r_id),
          *vtx_l = NULL;

    if( !edge_in_boundary) vtx_l = mesh->getNodePtr(vtx_l_id);

    // testes para identificar se algum dos vertices está bloqueado
    Real coords_t[3], coords_b[3];
    Real coords_m[3], t_aux=t;

    vtx_t->getCoord(coords_t, sdim);
    vtx_b->getCoord(coords_b, sdim);

    bool collapse_to_t = is_in(vtx_t->getTag(),slipvel_tags)||is_in(vtx_t->getTag(),flusoli_tags);
    bool collapse_to_b = is_in(vtx_b->getTag(),slipvel_tags)||is_in(vtx_b->getTag(),flusoli_tags);

    //if( (vtx_t->isBlocked()) && (!vtx_b->isBlocked()) )
    if (collapse_to_t)
    {
        t_aux=1.0;
    }
    //if( (!vtx_t->isBlocked()) && (vtx_b->isBlocked()) )
    if (collapse_to_b)
    {
        t_aux=0.0;
    }
    if(  (vtx_t->isBlocked()) && (vtx_b->isBlocked()) &&
         ((vtx_b->getTag()!=vtx_t->getTag()) || ((vtx_b->getTag()==vtx_t->getTag())&&(vtx_b->getTag()!=mesh->getFacetPtr(edge_m_id)->getTag())) ) ) return -1;

    for (int i = 0; i < sdim; ++i)
        coords_m[i] = t_aux*(coords_t[i] - coords_b[i]) + coords_b[i];

    vtx_t->setCoord(coords_m,sdim);
    vtx_b->setCoord(coords_m,sdim);

    // ================================ Pegar a estrela do vertice que vai ser substituido =============================
    int iCs_t[64],   // Ids das celulas da estrela
        viCs_t[64],
        iCs_b[64],
        viCs_b[64];  // Ids locais do vertice nas celulas da estrela

    mesh->vertexStar(cell_A_id, face_Am_id , iCs_t, viCs_t);
    mesh->vertexStar(cell_A_id, (face_Am_id+1)%3 , iCs_b, viCs_b);

    // testa se o colapso gera elementos invertidos
    bool test=false;
    for( int i=0; iCs_t[i]!=-1; i++)
    {
        if((iCs_t[i]!=cell_A_id)&&(iCs_t[i]!=cell_B_id)&&(area_s(mesh->getCellPtr(iCs_t[i]), mesh) < 1e-8) ) test = true;
    }
    for( int i=0; iCs_b[i]!=-1; i++)
    {
        if((iCs_b[i]!=cell_A_id)&&(iCs_b[i]!=cell_B_id)&&(area_s(mesh->getCellPtr(iCs_b[i]), mesh) < 1e-8) ) test = true;
    }
    if(test == true)
    {
        vtx_t->setCoord(coords_t, sdim);
        vtx_b->setCoord(coords_b, sdim);
        return -1;
    }

    // Indica a todas as celulas que continham vtx_t que vtx_b está na posição agora
    for( int i=0; iCs_t[i] > -1; i++)
    {
       Cell *cell = mesh->getCellPtr(iCs_t[i]);
       cell->setNodeId(viCs_t[i], vtx_b_id);
    }

    // =========================== Atribuir as adjacencias aos vertices ================================================
    // vtx_t : será removido, logo não é necessário atualizar suas adjacencias
    // Vertice vtx_r
    if( vtx_r->getIncidCell() == cell_A_id )
    {
        int cell_incid = -1, cell_pos = -1;
        if( cell_AT_id > -1 )
        {
            cell_incid  = cell_AT_id;
            cell_pos    = (face_ATtr_id+1)%3;
        }
        else if( cell_AB_id > -1 )
        {
            cell_incid  = cell_AB_id;
            cell_pos    = face_ABbr_id;
        }
        vtx_r->setIncidence(cell_incid, cell_pos);
    }

    // Vertice vtx_b
    if( (vtx_b->getIncidCell() == cell_A_id) || (vtx_b->getIncidCell() == cell_B_id) )
    {
        int cell_incid = -1, cell_pos = -1;
        if( cell_AT_id > -1 )
        {
            cell_incid  = cell_AT_id;
            cell_pos    = face_ATtr_id;
        }
        else if( cell_AB_id > -1 )
        {
            cell_incid  = cell_AB_id;
            cell_pos    = (face_ABbr_id+1)%3;
        }
        else if( cell_BB_id > -1 )
        {
            cell_incid  = cell_BB_id;
            cell_pos    = face_BBbl_id;
        }
        else if( cell_BT_id > -1 )
        {
            cell_incid  = cell_BT_id;
            cell_pos    = (face_BTtl_id+1)%3;
        }

        vtx_b->setIncidence(cell_incid, cell_pos);
    }

    // Vertice vtx_l
    if ( !edge_in_boundary )
    {
        if( vtx_l->getIncidCell() == cell_B_id )
        {
            int cell_incid = -1, cell_pos = -1;
            if( cell_BT_id > -1 )
            {
                cell_incid  = cell_BT_id;
                cell_pos    = face_BTtl_id;
            }
            else if( cell_BB_id > -1 )
            {
                cell_incid  = cell_BB_id;
                cell_pos    = (face_BBbl_id+1)%3;
            }
            vtx_l->setIncidence(cell_incid, cell_pos);
        }
    }

    // =========================== Atualizar as tags dos vertices ================================================
    if (mesh->inBoundary(vtx_t)||(vtx_t->isBlocked()))
    {
        int tag = vtx_t->getTag();
        vtx_b->setTag(tag);
    }
    else
    {
        int tag_t = vtx_t->getTag();
            //tag_b = vtx_b->getTag();
        //if( tag_t > tag_b )
        //    vtx_b->setTag(tag_t);
        if (collapse_to_t)
          vtx_b->setTag(tag_t);
    }

    if(vtx_t->isBlocked())
    {
        vtx_b->setBlockedTo(true);
    }

    /* old
    if( cell_AT_id > -1 )
    {
        if(! vtx_b->replacesIncidCell(cell_A_id, cell_AT_id, face_ATtr_id))
        {
            if( cell_BT_id > -1 )
                vtx_b->replacesIncidCell(cell_B_id, cell_BT_id, (face_BTtl_id+1)%3 );
            else
                vtx_b->replacesIncidCell(cell_B_id, cell_BB_id, (face_BBbl_id) );
        }

        vtx_r->replacesIncidCell(cell_A_id, cell_AT_id, (face_ATtr_id+1)%3 );
    }else
    {
        if(! vtx_b->replacesIncidCell(cell_A_id, cell_AB_id, (face_ABbr_id+1)%3))
        {
            if( cell_BT_id > -1 )
                vtx_b->replacesIncidCell(cell_B_id, cell_BT_id, (face_BTtl_id+1)%3);
            else
                vtx_b->replacesIncidCell(cell_B_id, cell_BB_id, (face_BBbl_id)%3 );
        }
        vtx_r->replacesIncidCell(cell_A_id, cell_AB_id, (face_ABbr_id) );
    }

    if ( !edge_in_boundary )
        vtx_l->replacesIncidCell(cell_B_id, cell_BT_id, face_BTtl_id);
    */
    // =========================== Atribuir as adjacencias das arestas =================================================

    Facet *edge_br = mesh->getFacetPtr(edge_br_id),
          *edge_tr = mesh->getFacetPtr(edge_tr_id);
    Facet *edge_bl = NULL,
          *edge_tl = NULL;

    // Aresta edge_br
    if(cell_AT_id > -1 ) edge_br->setIncidence(cell_AT_id, face_ATtr_id );
    else                 edge_br->setIncidence(cell_AB_id, face_ABbr_id );
    if(edge_tr->isBlocked()) edge_br->setBlockedTo(true);

    // Aresta edge_bl
    if ( !edge_in_boundary )
    {
        edge_bl=mesh->getFacetPtr(edge_bl_id);
        edge_tl=mesh->getFacetPtr(edge_tl_id);
        if(cell_BT_id>-1) edge_bl->setIncidence(cell_BT_id, face_BTtl_id);
        else              edge_bl->setIncidence(cell_BB_id, face_BBbl_id);
        if(edge_tl->isBlocked()) edge_bl->setBlockedTo(true);
    }
/*
    if (collapse_to_t){
      VectorXi edge_nodes(3); cout << endl;// 3 nodes at most
      mesh->getFacetNodesId(edge_br_id, edge_nodes.data());
      cout << "edge_br: " << edge_br->getTag() << ", nodes: " << edge_nodes[0] << "," << edge_nodes[1] << endl;
      mesh->getFacetNodesId(edge_tr_id, edge_nodes.data());
      cout << "edge_tr: " << edge_tr->getTag() << ", nodes: " << edge_nodes[0] << "," << edge_nodes[1] << endl;
      mesh->getFacetNodesId(edge_bl_id, edge_nodes.data());
      cout << "edge_bl: " << edge_bl->getTag() << ", nodes: " << edge_nodes[0] << "," << edge_nodes[1] << endl;
      mesh->getFacetNodesId(edge_tl_id, edge_nodes.data());
      cout << "edge_tl: " << edge_tl->getTag() << ", nodes: " << edge_nodes[0] << "," << edge_nodes[1] << endl;
    }
*/
    // =========================== Atualizar as tags das arestas =======================================================
    if (mesh->inBoundary(edge_tr) || edge_tr->isBlocked())
    {
        int tag = edge_tr->getTag();
        edge_br->setTag(tag);
    }
    else
    {
        //int tag_b = edge_br->getTag(),
        //    tag_t = edge_tr->getTag();
        //if( tag_t > tag_b )
        //    edge_br->setTag(tag_t);
      if (collapse_to_b)
        edge_br->setTag(vtx_r->getTag());
      //else
      //  if (edge_tr->getTag() != edge_br->getTag())
      //    edge_br->setTag(edge_tr->getTag());
    }

    if ( !edge_in_boundary )
    {
        if( mesh->inBoundary(edge_tl) || edge_tl->isBlocked() )
        {
            int tag = edge_tl->getTag();
            edge_bl->setTag(tag);
        }
        else
        {
            //int tag_b = edge_bl->getTag(),
            //    tag_t = edge_tl->getTag();
            //if( tag_t > tag_b )
            //{
            //    edge_bl->setTag(tag_t);
            //}
          if (collapse_to_b)
            edge_bl->setTag(vtx_l->getTag());
          //else
          //  if (edge_tl->getTag() != edge_bl->getTag())
          //    edge_bl->setTag(edge_tl->getTag());
        }
    }

    // ======= Fala para as celulas vizinhas o id das arestas que são suas faces, e quais são suas celulas vizinhas ====
    // =======================================  ----- SetUp Neighbors --------------  ==================================

    if ( cell_AT_id > -1 )
    {
        Cell *cell_AT = mesh->getCellPtr(cell_AT_id);
        cell_AT->setIncidence(face_ATtr_id, cell_AB_id, face_ABbr_id);
        cell_AT->setFacetId(face_ATtr_id, edge_br_id);
    }
    if ( cell_AB_id > -1 )
    {
        Cell *cell_AB = mesh->getCellPtr(cell_AB_id);
        cell_AB->setIncidence(face_ABbr_id, cell_AT_id, face_ATtr_id);
    }

    if (!edge_in_boundary)
    {
        if(cell_BT_id > -1 )
        {
            Cell *cell_BT = mesh->getCellPtr(cell_BT_id);
            cell_BT->setFacetId(face_BTtl_id, edge_bl_id);
            cell_BT->setIncidence(face_BTtl_id, cell_BB_id, face_BBbl_id);
        }
        if(cell_BB_id > -1 )
        {
            Cell *cell_BB = mesh->getCellPtr(cell_BB_id);
            cell_BB->setIncidence(face_BBbl_id, cell_BT_id, face_BTtl_id);
        }
    }

    // ========================================== Caso de vtx_t estar no bordo ===========================================
    // Se vtx_t esta no bordo, vtx_b tambem estara
    if (mesh->inBoundary(vtx_t))
        vtx_b->setAsBoundary(true);

    // ================================ Deletar os elementos ===========================================================
    mesh->disablePoint(vtx_t_id);

    mesh->disableFacet(edge_tr_id);
    mesh->disableFacet(edge_m_id);
    if(!edge_in_boundary)
        mesh->disableFacet(edge_tl_id);

    if (false && collapse_to_t){
      mesh->disableFacet(edge_br_id);
      mesh->disableFacet(edge_m_id);
      if(!edge_in_boundary)
          mesh->disableFacet(edge_bl_id);
    }

    mesh->disableCell(cell_A_id);
    if(!edge_in_boundary)
        mesh->disableCell(cell_B_id);

    // Casos degenerados
    if( ( cell_AT_id == -1 )&&( cell_AB_id ==-1 ) )
    {
        mesh->disableFacet(edge_br_id);
        mesh->disablePoint(vtx_r_id);
    }
    if( ( cell_BT_id == -1 )&&( cell_BB_id ==-1 )&&(edge_bl_id!=-1))
    {
        mesh->disableFacet(edge_bl_id);
        mesh->disablePoint(vtx_l_id);
    }

    return vtx_b_id;
}

Real AppCtx::area_s(Cell const* cell, Mesh const* mesh)
{
    Point const *a = mesh->getNodePtr(cell->getNodeId(0));
    Point const *b = mesh->getNodePtr(cell->getNodeId(1));
    Point const *c = mesh->getNodePtr(cell->getNodeId(2));

    Real const ax = a->getCoord(0), ay = a->getCoord(1);
    Real const bx = b->getCoord(0), by = b->getCoord(1);
    Real const cx = c->getCoord(0), cy = c->getCoord(1);

    Real At = 0.5 * ( ax*by + ay*cx + bx*cy - by*cx - ay*bx - ax*cy );

    return At;
}

void AppCtx::getTangents(Vector& T, Vector &B, Vector const& N, int dim){
  Tensor I(dim,dim);
  I.setIdentity();
  T.setZero();
  B.setZero();

  if (dim == 2){
    T(0) = +N(1); T(1) = -N(0);  //T(0) = -N(1); T(1) = +N(0);
  }
  else if (dim == 3){//TODO
    Vector Ref(3);
    Ref(0) = 0.0; Ref(1) = 0.0; Ref(2) = -1.0;
    Vector Tg = (I - N*N.transpose())*Ref;
    double nTg = Tg.norm();
    if (nTg < 1e-10){Tg(0) = 1.0; Tg(1) = 0.0; Tg(2) = 0.0;}
    else {Tg = Tg/nTg;}

    T = Tg;
    //cross(T2, N, T1);
  }
}
