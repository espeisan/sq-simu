#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

  bool help_return;
  bool erro;
  AppCtx user(argc, argv, help_return, erro);

  if (help_return)
    return 0;

  if (erro)
    return 1;

#ifdef FEP_HAS_OPENMP
  {
    int nthreads = 0;
    #pragma omp parallel
    {
      #pragma omp critical
      nthreads = omp_get_num_threads();
    }
    printf("OpenMP version.\n");
    printf("Num. threads: %d\n", nthreads);
  }
#else
  {
    printf("Serial version.\n");
  }
#endif


  user.loadMesh();
  user.loadDofs();  //counts dofs for U,P,V in cell, facet, corner
  user.evaluateQuadraturePts();

  erro = user.err_checks(); if (erro) return 1;

  // print info
  cout << endl; cout << "mesh: " << user.filename << endl;
  user.mesh->printInfo();
  cout << "\n# velocity unknowns: " << user.n_unknowns_u;
  cout << "\n# pressure unknowns: " << user.n_unknowns_p << " = "
                                    << user.n_unknowns_u/user.dim << " + " << user.n_nodes_fsi << " + " << user.n_nodes_sv;
//  cout << "\n# total unknowns: " << user.n_unknowns << endl;
  if (user.N_Solids){
    cout << "\n# velocity fluid only unknowns: " << user.n_unknowns_u - user.dim*(user.n_nodes_so+user.n_nodes_fsi+user.n_nodes_sv);
    cout << "\n# velocity solid only unknowns: " << user.n_nodes_so*user.dim;
    cout << "\n# velocity solid intf unknowns: " << user.n_unknowns_z << " (" <<
                                                   (user.n_nodes_fsi+user.n_nodes_sv) << ") = " <<
                                                   user.n_unknowns_z*(user.n_nodes_fsi+user.n_nodes_sv);
    cout << "\n# total unknowns: " << user.n_unknowns_fs << " = " << user.n_unknowns_fs - user.dim*(user.n_nodes_fsi+user.n_nodes_sv)
                                                         << " + " << user.dim*(user.n_nodes_fsi+user.n_nodes_sv);
    cout << "\n# solids: " << user.N_Solids << endl;
    cout << "unknowns distribution: " << 0 << "-" << user.n_unknowns_u-1 <<
            ", " << user.n_unknowns_u << "-" <<
            user.n_unknowns_u + user.n_unknowns_p - 1 <<
            ", " << user.n_unknowns_u + user.n_unknowns_p <<
            "-"  << user.n_unknowns_u + user.n_unknowns_p +
                    user.n_unknowns_z - 1;
  }
  cout << endl;
  user.mesh->printStatistics();
  user.mesh->timer.printTimes();

  cout << endl;
  if (user.N_Solids){
    cout << "flusoli_tags =  ";
    for (int i = 0; i < (int)user.flusoli_tags.size(); ++i)
    {
      cout << user.flusoli_tags[i] << " ";
    }
  cout << endl;
  }

  printf("on update mesh\n");
  user.onUpdateMesh();
  user.solveTimeProblem();

  cout << "\n";
  user.timer.printTimes();

  user.freePetscObjs();
  PetscFinalize();

  cout << "\a" << endl; cout << "\a" << endl;
  return 0.;
}
