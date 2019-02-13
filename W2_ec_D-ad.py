"""Python file written to run on CX1, producing Williamson2 test case
simulation, with energy conserving space discretisation including
upwinding for u, D, and Poisson-type time discretisation"""
from time import ctime
from numpy import float64, zeros, arctan2, arcsin
from numpy import sqrt as np_sqrt
from netCDF4 import Dataset
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, exp, \
    as_vector, pi, FunctionSpace, FiniteElement, MixedFunctionSpace, \
    Function, TestFunction, TrialFunction, inner, dx, triangle, grad, \
    LinearVariationalProblem, LinearVariationalSolver, File, lhs, rhs, \
    FacetNormal, jump, sign, dot, dS, TestFunctions, CellNormal, cross, \
    TrialFunctions, div, assemble, sqrt, conditional, Constant, \
    DumbCheckpoint, FILE_READ, FILE_CREATE, COMM_WORLD, \
    Min, Max, atan_2, asin, cos, sin, VectorFunctionSpace, Mesh, \
    functionspaceimpl, READ, WRITE, par_loop, op2, ge, le, interpolate

# discretisation parameters
ref_level = 5
dt = 50.
tmax = 24*60*60*50.
init_t = 0.
maxk = 4
dumpnumber = 4
nc_h5_dumpfreq = 600
pickup = False

fname = 'W2_ec_D-ad'
write_latlon = True

nc_diag = {'Energy': 0., 'Enstrophy': 0., 'Mass': 0.,
           'L2_error_u': 0, 'L2_error_D': 0}
h5_safe, nc_safe = True, True

if COMM_WORLD.Get_rank() == 0:
    print("Simulation for Williamson2 test case, model u-ad_D-ad", "\n",
          "| Discr details: Poisson implicit, no EP, TM=0.5", "\n",
          "| dt", dt, "| tmax", tmax, " | refinement level", ref_level,
          "\n", "| maxk", maxk, "| number of dumps (exlcuding t=0):",
          dumpnumber, "| nc, h5 dump frequency", nc_h5_dumpfreq, "\n",
          "| pickup", pickup, "| Diagnostics:", tuple(nc_diag.keys()))
    print("Starting Initial condition, and function setup at", ctime())

R, Omega, = 6371220., 7.292e-5
mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref_level,
                             degree=2)
mesh.init_cell_orientations(SpatialCoordinate(mesh))

x = SpatialCoordinate(mesh)
f = Function(FunctionSpace(mesh, "CG", 1))
f.interpolate(2*Omega*x[2]/R)
g, H = 9.810616, 5960.
u_0 = 2*pi*R/(12*24*60*60.)

uexpr = u_0*as_vector([-x[1], x[0], 0.0])/R
Dexpr = H - (R*Omega*u_0+u_0**2/2.)*x[2]**2/(g*R**2)
bexpr = Constant(0.)

# Build function spaces
degree = 2
family = ("DG", "BDM", "CG")
W0 = FunctionSpace(mesh, family[0], degree-1, family[0])
W1_elt = FiniteElement(family[1], triangle, degree)
W1 = FunctionSpace(mesh, W1_elt, name="HDiv")
W2 = FunctionSpace(mesh, family[2], degree+1)
M = MixedFunctionSpace((W1, W0))

# Set up functions
xn = Function(M)
un, Dn = xn.split()
un.rename('u')
Dn.rename('D')
fields = {'u': un, 'D': Dn}

# Energy field for output
En = Function(W0, name='Energy')
e_form = lambda u, D: 0.5*(D*inner(u, u) + g*(D + b)**2)

# Vorticity field
qn = Function(W2, name='potential vorticity')
fields['q'] = qn
# Vorticity field
vortn = Function(W2, name='vorticity')
# Enstrophy field
q2Dn = Function(W2, name='enstrophy')
# Topography field
b = Function(W0, name='topography')
eta_out = Function(W0, name='eta')

# Solver fields
uad = Function(W1)
uf = Function(W1)

# Time scheme fields
xp = Function(M)
up, Dp = xp.split()
qp = Function(W2)

xnk = Function(M)
unk, Dnk = xnk.split()

ubar = 0.5*(un + unk)
Dbar = 0.5*(Dn + Dnk)

xd = Function(M)

# Hamiltonian variations
F = Function(W1)
P = Function(W0)
u_rec = Function(W1)

# Load initial conditions onto fields and save them for error diagnostic
un.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})
Dn.interpolate(Dexpr)
b.interpolate(bexpr)

u0, D0, q0 = Function(W1), Function(W0), Function(W2)
u_err = Function(W1, name='u error')
D_err = Function(W0, name='D error')
q_err = Function(W2, name='pv error')
u0.assign(un)
D0.assign(Dn)

if pickup:
    with DumbCheckpoint("chkpt_{0}".format(fname), mode=FILE_READ) as chk:
        chk.load(xn)
        init_t = chk.read_attribute("/", "time")


if COMM_WORLD.Get_rank() == 0:
    print("Finished setting up functions at", ctime())

# Build perp and upwind perp (including 2D version for test purposes)
n = FacetNormal(mesh)
s = lambda u: 0.5*(sign(dot(u, n)) + 1)
uw = lambda u, v: (s(u)('+')*v('+') + s(u)('-')*v('-'))

if mesh.geometric_dimension() == 2:
    perp = lambda u: as_vector([-u[1], u[0]])
    p_uw = lambda u, v: perp(uw(u, v))
else:
    perp = lambda u: cross(CellNormal(mesh), u)
    out_n = CellNormal(mesh)
    p_uw = lambda u, v: (s(u)('+')*cross(out_n('+'), v('+'))
                         +s(u)('-')*cross(out_n('-'), v('-')))

# Initial solve for vorticity for output
eta = TestFunction(W2)
q_ = TrialFunction(W2)
q_eqn = eta*q_*Dn*dx + inner(perp(grad(eta)), un)*dx - eta*f*dx
q_p = LinearVariationalProblem(lhs(q_eqn), rhs(q_eqn), qn)
q_solver = LinearVariationalSolver(q_p, solver_parameters=
                                   {"ksp_type":"preonly",
                                    "pc_type":"lu"})
q_solver.solve()
q0.assign(qn)

# Build advection, forcing forms
M_ext = MixedFunctionSpace((W1, W0, W0))
xp_ext = Function(M_ext)

u_, D_, P_ = TrialFunctions(M_ext)
v, psi, chi = TestFunctions(M_ext)

D_bar = 0.5*(Dn + D_)
u_bar = 0.5*(un + u_)

Frhs = unk*Dnk/3. + un*Dnk/6. + unk*Dn/6. + un*Dn/3.
K = inner(un, un)/3. + inner(un, unk)/3. + inner(unk, unk)/3.
Prhs = g*(D_bar + b) + 0.5*K

D_ad = (inner(grad(psi), D_bar*u_rec)*dx
        - jump(psi*u_rec, n)*uw(ubar, D_bar)*dS)
u_ad = (inner(perp(grad(inner(Dbar*v, perp(u_rec)))), u_bar)*dx
        + inner(jump(inner(Dbar*v, perp(u_rec)), n),
                p_uw(ubar, u_bar))*dS)
u_f = (jump(P_*v, n)*uw(ubar, Dbar)*dS - inner(Dbar*v, grad(P_))*dx
       - f*inner(perp(u_rec), Dbar*v)*dx)

af_eqn = ((D_ - Dn)*psi*dx - dt*D_ad
          + inner(u_ - un, Dbar*v)*dx - dt*(u_ad  + u_f)
          + inner(chi, P_ - Prhs)*dx)

af_p = LinearVariationalProblem(lhs(af_eqn), rhs(af_eqn), xp_ext)
af_solver = LinearVariationalSolver(af_p, solver_parameters=
                                    {"mat_type": "aij",
                                     "snes_linesearch_type": "basic",
                                     "ksp_type": "preonly",
                                     "pc_type": "lu"})

# Auxiliary solvers
u_ = TrialFunction(W1)
w = TestFunction(W1)
u_rec_eqn = inner(w, Dbar*u_ - Frhs)*dx
u_rec_problem = LinearVariationalProblem(lhs(u_rec_eqn), rhs(u_rec_eqn),
                                         u_rec)
u_rec_solver = LinearVariationalSolver(u_rec_problem, solver_parameters=
                                       {"ksp_type":"preonly",
                                        "pc_type":"lu"})

# Output solvers
q_eqn = eta*q_*Dnk*dx + inner(perp(grad(eta)), unk)*dx - eta*f*dx
q_problem = LinearVariationalProblem(lhs(q_eqn), rhs(q_eqn), qp)
qsolver = LinearVariationalSolver(q_problem, solver_parameters=
                                  {"ksp_type":"preonly",
                                   "pc_type":"lu"})

vrt_eqn = eta*q_*dx + inner(perp(grad(eta)), un)*dx
vort_problem = LinearVariationalProblem(lhs(vrt_eqn), rhs(vrt_eqn), vortn)
vortsolver = LinearVariationalSolver(vort_problem, solver_parameters=
                                     {"ksp_type":"preonly",
                                      "pc_type":"lu"})

# Linear solver
u, D = TrialFunctions(M)
w, phi = TestFunctions(M)

eqn = (inner(w, u - (up - unk))
       +phi*(D - (Dp - Dnk))
       -0.5*dt*(g*div(w)*D - f*inner(w, perp(u)) - H*phi*div(u)))*dx

params = {'mat_type': 'matfree',
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.HybridizationPC',
          'hybridization': {'ksp_type': 'preonly',
                            'pc_type': 'lu',
                            'pc_factor_mat_solver_type': 'mumps'}}

uD_problem = LinearVariationalProblem(lhs(eqn), rhs(eqn), xd)
uD_solver = LinearVariationalSolver(uD_problem, solver_parameters=params)

if COMM_WORLD.Get_rank() == 0:
    print("Finished setting up solvers at  ", ctime())

# Setup output
outfile = File('{0}.pvd'.format(fname))
field_output = [un, Dn, u_err, D_err, q_err]
field_dumptimes = [i*int(tmax/(dumpnumber)) for i in range(dumpnumber+1)]

# Create latlon version of output
def get_latlon_mesh(mesh):
    """Build 2D projected mesh of spherical mesh"""
    crds_orig = mesh.coordinates
    mesh_dg_fs = VectorFunctionSpace(mesh, "DG", 1)
    crds_dg = Function(mesh_dg_fs)
    crds_latlon = Function(mesh_dg_fs)
    par_loop("""
for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
        dg[i][j] = cg[i][j];
    }
}
""", dx, {'dg': (crds_dg, WRITE),
          'cg': (crds_orig, READ)})

    # lat-lon 'x' = atan2(y, x)
    crds_latlon.dat.data[:, 0] = arctan2(crds_dg.dat.data[:, 1],
                                         crds_dg.dat.data[:, 0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    crds_latlon.dat.data[:, 1] = (arcsin(crds_dg.dat.data[:, 2]/
                                         np_sqrt(crds_dg.dat.data[:, 0]**2
                                                 +crds_dg.dat.data[:, 1]**2
                                                 +crds_dg.dat.data[:, 2]**2
                                                 )))
    crds_latlon.dat.data[:, 2] = 0.0

    kernel = op2.Kernel("""
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586
void splat_coords(double **coords) {
    double diff0 = (coords[0][0] - coords[1][0]);
    double diff1 = (coords[0][0] - coords[2][0]);
    double diff2 = (coords[1][0] - coords[2][0]);

    if (fabs(diff0) > PI || fabs(diff1) > PI || fabs(diff2) > PI) {
        const int sign0 = coords[0][0] < 0 ? -1 : 1;
        const int sign1 = coords[1][0] < 0 ? -1 : 1;
        const int sign2 = coords[2][0] < 0 ? -1 : 1;
        if (sign0 < 0) {
            coords[0][0] += TWO_PI;
        }
        if (sign1 < 0) {
            coords[1][0] += TWO_PI;
        }
        if (sign2 < 0) {
            coords[2][0] += TWO_PI;
        }
    }
}
""", "splat_coords")

    op2.par_loop(kernel, crds_latlon.cell_set,
                 crds_latlon.dat(op2.RW, crds_latlon.cell_node_map()))
    return Mesh(crds_latlon)


if write_latlon:
    outfile_ll = File('{0}_ll.pvd'.format(fname))
    field_output_ll = []
    mesh_ll = get_latlon_mesh(mesh)
    for f in field_output:
        FSpwG = functionspaceimpl.WithGeometry
        field_ll = Function(FSpwG(f.function_space(), mesh_ll),
                            val=f.topological, name=f.name() + '_ll')
        field_output_ll.append(field_ll)
else:
    outfile_ll = None
    field_output_ll = None

nc_temp = zeros((nc_h5_dumpfreq, len(nc_diag)))

# Create checkpoint
chkpt = DumbCheckpoint("chkpt_{0}".format(fname), mode=FILE_CREATE)
if h5_safe:
    chkpt_bckp = DumbCheckpoint("chkpt_bckp_{0}".format(fname),
                                mode=FILE_CREATE)

# Nc file functions
def create_nc_file(pfx=''):
    """diagnostics nc file setup function"""
    with Dataset('diagnostics{0}_{1}.nc'.format(pfx, fname), "w") as ds:
        # Create dataset description
        ds.description = ("Total amount of energy for RSWE model")
        ds.history = "Created {t}".format(t=ctime())
        ds.createDimension("time", None)
        # Build time variable
        var = ds.createVariable("time", float64, ("time", ))
        var.units = "seconds"
        # Add nc_diagnostics
        for k in nc_diag:
            group = ds.createGroup(k)
            group.createVariable("total", float64, ("time", ))


def write_to_nc_file(t, nc_array, pfx=''):
    """function to write to diagnostics nc file"""
    with Dataset('diagnostics{0}_{1}.nc'.format(pfx, fname), "a") as ds:
        for j in range(0, nc_h5_dumpfreq):
            idx = ds.dimensions["time"].size
            ds.variables["time"][idx:idx + 1] = t - (nc_h5_dumpfreq + j)*dt
            _c = 0
            for k in nc_diag:
                var = ds.groups[k].variables["total"]
                var[idx:idx + 1] = nc_array[j][_c]
                _c += 1

# Create nc files
if COMM_WORLD.Get_rank() == 0 and not pickup:
    create_nc_file()
    if nc_safe:
        create_nc_file('_bckp')

# output function
def write_output(diag, t, counter, dfr, nc_h5_dfr,
                 outf, outf_ll, fld_out, fld_out_ll):
    """Function to write vtu, txt output"""

    # Update and output diagnostics
    energy = e_form(un, Dn)
    En.interpolate(energy)
    if 'Energy' in diag.keys():
        diag['Energy'] = assemble(energy*dx)
    if 'Mass' in diag.keys():
        diag['Mass'] = assemble(Dn*dx)
    if 'Enstrophy' in diag.keys():
        diag['Enstrophy'] = assemble(qn**2*Dn*dx)
    if 'L2_error_D' in diag.keys():
        diag['L2_error_D'] = sqrt(assemble((Dn - D0)*(Dn - D0)*dx))
    if 'L2_error_u' in diag.keys():
        diag['L2_error_u'] = sqrt(assemble(inner(un - u0, un - u0)*dx))

    _c = 0
    for k in diag:
        nc_temp[counter % nc_h5_dfr][_c] = diag[k]
        _c += 1

    # Save I/O time by not writing at each timestep
    if (counter % nc_h5_dfr) == nc_h5_dfr - 1:

        if COMM_WORLD.Get_rank() == 0:
            print("Timestep nr", counter, ": Writing nc files at", ctime())
            write_to_nc_file(t, nc_temp)
            if nc_safe:
                write_to_nc_file(t, nc_temp, '_bckp')

        # Write to checkpoint
        chkpt.store(xn)
        chkpt.write_attribute("/", "time", t)
        if h5_safe:
            chkpt_bckp.store(xn)
            chkpt_bckp.write_attribute("/", "time", t)

    # Output vtu file
    if t in field_dumptimes:
        #q2Dn.project(qn**2*Dn)
        #eta_out.interpolate(Dn + b)
        #vortsolver.solve()
        u_err.assign(un - u0)
        D_err.assign(Dn - D0)
        q_err.assign(qn - q0)
        outf.write(*fld_out)
        if write_latlon:
            outf_ll.write(*fld_out_ll)


write_output(nc_diag, init_t, 0, field_dumptimes, nc_h5_dumpfreq,
             outfile, outfile_ll, field_output, field_output_ll)

if COMM_WORLD.Get_rank() == 0:
    print("Finished setting up output at   ", ctime())

# Timeloop
xnk.assign(xn)
t = init_t
count = 0
while t < tmax - 0.5*dt:
    t += dt

    # Run solvers
    for _ in range(maxk):
        u_rec_solver.solve()
        af_solver.solve()
        up.assign(xp_ext.split()[0])
        Dp.assign(xp_ext.split()[1])

        uD_solver.solve()
        xnk += xd

    xn.assign(xnk)
    qsolver.solve()
    qn.assign(qp)

    # Write output
    count += 1
    write_output(nc_diag, t, count, field_dumptimes, nc_h5_dumpfreq,
                 outfile, outfile_ll, field_output, field_output_ll)
