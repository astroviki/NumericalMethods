import numpy as np
import scipy.sparse as sp
import scipy.signal as sb

class Superparticles:
    """Class describing object Superparticle

    
    """
    
    def __init__(self,positions,velocities,charge,scale):
        """The Constructor for Superparticle object
        
        Parameters:
        position (numpy.array): Positions of particles 
        velocity (numpy.array): Velocities of particles 
        charge (float): Charge of Superparticle
        scale (float): Scale of kernel
        """
        
        self.number_of_particles = np.size(positions)
        self.positions = positions
        self.velocities = velocities
        self.forces  = np.zeros(self.number_of_particles)
        self.mass = 1
        self.charge = charge
        self.support_scale = scale
        return
    
    def set_positions(self,positions):
        """ Setting new position of Superparticle
        
        Parameters:
        positions (numpy.array): New x-positions of Superparticles 
        """
        
        self.positions = positions
        return 
    
    def set_velocities(self,velocities):
        """Setting new velocities of Superparticles
        
        Parameters:
        velocities (numpy.array): New x-velocities of Superparticles 
        """
        
        self.velocities = velocities
        return
    
    def get_positions(self):
        """Get positions of Superparticles 

        """
        return self.positions
    
    def get_velocities(self):
        """Get velocities of Superparticles 

        """
        return self.position
    
class Solver:
    """Class for solving PIC 1D electrostatic problem 

    """    
    
    def __init__(self,dt):
        """The Constructor for PIC 1D electrostatic solver 
        
        """
        self.dt = dt
        self.spline_order = 1
        self.epsilon = 1.0
        self.configuration={}
        self.configuration["min"] = 0.0
        self.configuration["max"] = 2.0*np.pi 
        self.configuration["points"] = 32
        self.number_of_superparticles = 10000
        self.order = 2
        self.box_length = (self.configuration["max"]-self.configuration["min"])
        self.deltax = self.box_length/self.configuration["points"]
        self.coeff=(self.deltax*self.deltax)/self.epsilon
        self.grid_x = self._create_grid()      
        self.grid_density = np.zeros((self.configuration["points"]))
        self.grid_efield = np.zeros((self.configuration["points"]))  
        self.v_amplitude = 0.01
        self.omega_pf = 1
        self.chargetomass = -1
        self.particlecharge = (self.omega_pf**2)/(self.chargetomass*self.number_of_superparticles/self.box_length)
        self.density_background= -self.number_of_superparticles*self.particlecharge/self.box_length
        return
    
    def _shape(self,x,y):
        """Private method calculate b-spline value for particles on x-position on grid

        Keyword arguments:
        ------------------
        x -- position on grid (numpy.array)
        y -- particle(s) position (numpy.array)
              
        Return:
        -------
        bspline (numpy.array of float) of B-spline(x,order) value representing kernel  
        """
        
        ksi = (x - y)/self.particles.support_scale
        return sb.bspline(ksi,self.spline_order)
    
    def _create_grid(self):
        """Private method create 1D spatial grid
        
        Return
        
        x_grid (numpy.array of float) dimension given by self.configuration["points"] attribute
        """
        return np.linspace(self.configuration["min"],self.configuration["max"],self.configuration["points"])
    
    def rightside_vector(self):
        """Public method calculate vector of right side of linear system from discretized version of
        Poisson equation

        """
        vector = - self.coeff*np.copy(self.grid_density[1:-1])
        # Periodic boundary condition
        vector[0] = vector[0]-self.potential[0]
        vector[-1] = vector[-1]-self.potential[-1]
        self.b = vector
        return
    
    def poisson_equation(self):
        """Method solve Poisson equations using implicit scheme

        """
        
        # Number of inner points
        N = self.configuration["points"]-2
        # Diagonals stored row-wise in matrix
        matrix_data = np.array([np.ones(N), -2.0*np.ones(N),np.ones(N)])
        # Spare tridiagonal matrix creation
        self.A = sp.spdiags(matrix_data, [-1,0,1], N, N).toarray()
        self.potential = np.linalg.solve(self.A,self.b)
        # Periodic boundary condition
        self.potential=np.insert(self.potential,0,self.potential[-2])
        self.potential=np.append(self.potential,self.potential[1])
        return
    
    def create_particles(self):
        """Method create particles in plasma box

        Return:
        -------
        particles -- Superparticles object
        """
        
        xp = np.linspace(0,self.box_length-self.box_length/self.number_of_superparticles,self.number_of_superparticles)
        vp = self.v_amplitude*np.random.randn(self.number_of_superparticles)
        
        self.particles = Superparticles(xp,vp,self.particlecharge,self.deltax)
        return
                              
    def density_on_grid(self):
        """Method calculate density on grid using sumation over superparticles and kernel function

        Keyword arguments:
        ------------------
        particles -- Superparticles object
        
        Return:
        -------
        density_mesh (numpy.array of float) Density calculated on mesh from Superparticles
        """
        
        for i in range(1,self.configuration["points"]-1):
            x = self.grid_x[i]
            self.grid_density[i] = np.sum(self.particles.charge/self.deltax*\
                             self._shape(x,self.particles.positions))+\
                             self.density_background 

        # Periodic boundary 
        self.grid_density[0]=self.grid_density[-2]
        self.grid_density[-1]=self.grid_density[1]
        return
    
    def electric_field_on_grid(self):
        """Method calculate intensity of electric field on grid using gradient 
            of electric potential from Poisson equation according to formula
            
             Ei = - grad( phi ) 
             
             using centered scheme
             
             E_i = - [ phi_(i+1) - phi_(i-1) ] / [2.0 delta(x)]
        """
        self.grid_efield[1:-2]= - (self.potential[2:-1]-self.potential[:-3])/(2.0*self.deltax)
        
        # Periodic boundary condition
        self.grid_efield[0] = self.grid_efield[-2]
        self.grid_efield[-1] = self.grid_efield[1]
        return
    
    def update_particlesposition(self):
        """Equation of motion solved by leapfrog method for advencing position of particles

        """
        self.particles.positions +=self.dt*self.particles.velocities

        return
    
    def update_particlesvelocity(self):
        """Equation of motion solved by leapfrog method for advencing veloicty of particles

        """

        self.particles.velocities+=self.dt*self.particles.forces
        return
    
    def update_particlesforce(self):
        """Electric field acting on Superparticles

        """
        for j,x_particle in enumerate(self.particles.positions):
            self.particles.forces[j] = np.sum(np.multiply(self.grid_efield,self._shape(self.grid_x,x_particle)))
        return
     
    def boundary_limits(self):
        """Boundary condition applied on Superparticles

        """
        L=self.box_length
        self.particles.positions[np.where(self.particles.positions >= L)] -= L
        self.particles.positions[np.where(self.particles.positions < 0)] += L
        
        return
    
    def initial_conditions(self):
        """
        Intial loading for 2-Stream plasma instability case
        """
        v_amplitude = 0.2
        mode = 1.0
        xp1 = 1.0
        vt=0.0
        v1=0.0
        ln = self.box_length/self.number_of_superparticles

        vp=vt*np.random.rand(self.number_of_superparticles)
        xp = np.linspace(0,self.box_length-ln, self.number_of_superparticles)

        pm = np.arange(self.number_of_superparticles)
        pm = -1+2*np.mod(pm,2)
        vp = vp+pm*v_amplitude
        
        self.particles.positions = xp+xp1*ln*np.sin(2.0*np.pi*xp/self.box_length*mode)
        self.particles.velocities = vp+v1*np.sin(2.0*np.pi*xp/self.box_length*mode)
        
        self.potential = np.zeros(self.configuration["points"])
        
        return