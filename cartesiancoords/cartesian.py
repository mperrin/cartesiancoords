
""" Cartesian Coordinate Systems

    This is mostly a wrapper built on top of the CoordSys python module
        http://pypi.python.org/pypi/CoordSys/0.52
        or http://download.j-raedler.de/Python/CoordSys/

    That module provides the basic machinery for fast transformations between coordinate systems.
    However, it has a fairly brittle interface and does not keep track of which system a vector is in.

    This module extends it to provide vector classes which are coordinate-system aware, i.e. each vector
    knows what its own coordinate system is. There are classes for two different types of vectors, coordinates
    and displacements, plus rotations. 

    Such vectors may be manipulated in various ways and transformed into any given coordinate system.
    Because each vector knows its own system, it is possible to add or subtract vectors in different
    coordinate systems and get the proper result, because the internal computation will transform them
    to a common system prior to the vector operation.



"""
import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import unittest

try:
    from IPython.Debugger import Tracer; stop = Tracer()
except:
    pass

import CoordSys


__all__= ['RotX','RotY','RotZ','CoordVector','DisplacementVector', 'CoordRotation', 'CoordAxes', 'CoordAxesGlobal']
_axes_list = {}

# convenience functions for math in degrees
cos_d = lambda x: np.cos(np.deg2rad(x))
sin_d = lambda x: np.sin(np.deg2rad(x))

toTuple = lambda thing: tuple(np.array(thing).flatten())

def RotX(rot_angle):
    """ Return a rotation matrix for a rotation around the X axis. 

    Parameters
    -----------
    rot_angle : float
        Rotation angle, in degrees

    Returns 
    ---------
    rotmat : numpy.matrix
        Rotation matrix
    """
    return np.matrix([[1,                0,                0],
                     [0, cos_d(rot_angle), -sin_d(rot_angle)], \
                     [0, sin_d(rot_angle),  cos_d(rot_angle)]])

def RotY(rot_angle):
    """ Return a rotation matrix for a rotation around the Y axis. 
    
    Parameters and Returns same as RotX
    """
    return np.matrix([[cos_d(rot_angle), 0, sin_d(rot_angle)],
                     [0,                1,          0       ],
                     [-sin_d(rot_angle),0, cos_d(rot_angle)]])

def RotZ(rot_angle):
    """ Return a rotation matrix for a rotation around the Z axis. 
    
    Parameters and Returns same as RotX
    """
    return np.matrix([[cos_d(rot_angle),-sin_d(rot_angle), 0],
                     [sin_d(rot_angle),  cos_d(rot_angle),0],
                     [0,                0,                1]])

#---------------

class _SystemVector(np.matrix):
    """ Base class for vector in a coordinate system.

    Do not use this class directly; use one of its derived subclasses

    See CoordVector and DisplacementVector.
    """

    # Subclassing ndarray is tricky for a variety of reasons. See
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(subtype, data, system=None, *value, **args):
        # note: the call to np.asarray() in the next line is crucial
        # because it converts data back into a vanilla base ndarray
        # and thus forgets its current array subtype. Without this
        # it becomes impossible to cast between CoordVector and DisplacementVector
        # derived types.
        obj = np.matrix.__new__(subtype,np.asarray(data), *value, **args)
        if system is None:
            system = CoordAxesGlobal()

        obj.system = system
        #print subtype, type(data), type(np.asarray(data)), type(obj)
        return obj

    def __init__(self, *value, **args):
        np.matrix.__init__(self, *value, **args)
        #self.system = system

        if np.mod(self.size, 3) != 0:
            raise ValueError("Input array must have length some multiple of 3 to create a %s object." % type(self))
        self.shape = (3, self.size/3)

    def __getitem__(self,key):
        # Accessing single elements should return scalars not 1x1 matrixes
        ans = np.matrix.__getitem__(self,key)
        if ans.size ==1:
            return float(ans)
        else:
            return ans

    def __array_finalize__(self, obj):
        # make sure to propagate coordinate system tag to children after array manipulations
        # get the system attribute from the parent object if at all possible, else use Global
        self.system = getattr(obj, 'system', CoordAxesGlobal())

    def toTuple(self):
        """ Convert a given vector to a tuple """
        # fixme - single vectors vs arrays of vectors here
        return tuple(np.array(self).flatten())

    def mag(self):
        """ Return the magnitude (length) of a vector """
        return np.sqrt(self[0]**2+self[1]**2+self[2]**2)


class CoordVector(_SystemVector):
    """ A vector class which knows what coordinate system it is in.

    A CoordVector represents a specific point in 3-dimensional space in some
    global coordinate system. When transferred into a different coordinate system,
    the magnitude and direction of this vector will change such that it points to
    the same absolute position in space.

    See also DisplacementVector

    Numerically, this is either a 3x1 numpy matrix (for one vector)
    or a 3xN numpy matrix (for multiple vectors)

    """
    __array_priority__ = 12.0

    def toGlobal(self):
        """Cast an arbitrary vector from its native coordinate system back to global coords.
        Returns a CoordVector in the global system. """
        if str(self.system.name)=='_global_':
            return self
        else:
            try:
                return self.system.toGlobal( (self.toTuple(),))
            except:
                raise ValueError("Something wrong with that coordinate system?")
    def toLocal(self, coordsys):
        """Cast an arbitrary vector from its native coordinates to some local coords.

        Returns a CoordVector in the requested local system. """

        if self.system == coordsys:
            return self # no need to change systems
        elif isinstance(coordsys, CoordAxesGlobal):
            return self.toGlobal()
        else:

            globalself = self.toGlobal() # first cast to global, in case we are already in some local system.
            try:
                return CoordVector(coordsys.toLocal( (globalself.toTuple(),)), system=coordsys)
            except:
                raise ValueError("Something wrong with that coordinate system?")

    def draw3d(self, ax=None, **kwargs):
        """ Plot a vector, in global coordinates 
        
        Parameters
        ------------
        ax : matplotlib.Axes, optional
            Axes to draw into. If not set, current axes are used. 
        """

        if ax is None:
            ax = pl.gca()

        arrowcoords = np.zeros((2,3))
        arrowcoords[0] = np.array(self.system.origin()).flatten()
        arrowcoords[1] = np.array(self.toGlobal()).flatten()
        ax.plot(arrowcoords[:,0].flatten(),arrowcoords[:,1].flatten(),arrowcoords[:,2].flatten(), zorder=200, **kwargs)
        pl.draw()

    def __sub__(self, other):
        """ Subtract two vectors"""
        if isinstance(other, DisplacementVector):
            # difference of two displacements is a Coord
            localdisp = other.toLocal(self.system)
            return CoordVector( np.matrix.__sub__(self, localdisp), system=self.system)
        elif isinstance(other, CoordVector ):
            # difference of two coords is a Displacement
            localother = other.toLocal(self.system)
            return DisplacementVector( np.matrix.__sub__(self,localother), system=self.system )
        else:
            raise TypeError("Can onlysubtract a  CoordVector or DisplacementVector from a CoordVector")

    def __add__(self, other):
        """ Add two vectors"""
        if not isinstance(other, DisplacementVector):
            raise TypeError("Can only add a DisplacementVector to a CoordVector")

        localdisp = other.toLocal(self.system)
        return CoordVector( np.matrix.__add__(self, localdisp), system=self.system)

    def __eq__(self,other):
        """ compare two vectors """
        if not isinstance(other, CoordVector):
            raise TypeError("Can only compare CoordVectors with each other")

        other_local = other.toLocal(self.system)
        # test with some given precision?
        raise NotImplementedError('Not implemented yet... need to define acceptable precision for equality')
        stop()

    def __repr__(self):
        return "<CoordVector [%g, %g, %g] in system '%s'>" % (self[0], self[1], self[2], self.system.name)


class DisplacementVector(_SystemVector):
    """ A vector class which knows what coordinate system it is in.

    A DisplacementVector represents some motion or change in 3-dimensional space.
    When transferred into a different coordinate system, its magnitude will remain
    constant but the direction will change according to the axes conventions in that
    coordinate system such that the same displacement in global coordinates remains.
   

    Numerically, this is a 3x1 numpy matrix. (for one vector)


    See also
    ----------
    CoordVector
    """
    __array_priority__ = 13.0

 
    def __sub__(self, other):
        if not isinstance(other,DisplacementVector):
            raise TypeError("Can only subtract one DisplacementVector from another")
        localother = other.toLocal(self.system)

        return DisplacementVector( np.matrix.__sub__(self,localother), system=self.system )

    def __add__(self, other):
        """ Add two vectors"""
        if isinstance(other, DisplacementVector):
            # two displacements yields a displacement
            localdisp = other.toLocal(self.system)
            return DisplacementVector( np.matrix.__add__(self, localdisp), system=self.system)
        elif  isinstance(other, CoordVector):
            # a displacement and a coordinate yields a new coordinate
            # in the same system as the original coordinate
            localdisp = self.toLocal(other.system)
            return CoordVector( np.matrix.__add__(other, localdisp), system=other.system)
        else:
            raise TypeError("Can only add DisplacementVectors to other DisplacementVectors or CoordVectors")

    def __repr__(self):
        return "<Displacement [%f, %f, %f] in system '%s'>" % (self[0], self[1], self[2], self.system.name)

    def toLocal(self, coordsys):
        """Cast an arbitrary vector from its native coordinates to some local coords.

        Returns a DisplacementVector in the requested local system. """

        if self.system == coordsys:
            return self # no need to change systems
        elif isinstance(coordsys, CoordAxesGlobal):
            return self.toGlobal()
        else:
            # consider this vector in its original coordinate system. Define a start (origin) and
            # end (origin+displacement).  We can transfer both of those points to the new coordinate
            # system, and then their difference yields the displacement in the new coordinate system.
            origin_l = CoordVector([0,0,0], system=self.system) # origin in local coords
            point2_l = (origin_l + self)

            point1 = origin_l.toLocal(coordsys)
            point2 = point2_l.toLocal(coordsys)
            return point2 - point1
    def toGlobal(self):
        """Cast vector to the global coordinate system. """
        if self.system is CoordAxesGlobal():
            return self
        else:
            origin_g = self.system.origin() # origin always returns in global coords
            origin_l = CoordVector([0,0,0], system=self.system) # origin in local coords
            point2_l = (origin_l + self)
            point2_g = point2_l.toGlobal()
            return point2_g - origin_g

class CoordRotation(object):
    """
    Coordinate-system independent rotation conversion class, which nonetheless
    knows what coordinate system it is referenced to.

    This represents the rotation of some object about the origin of *that specified coordinate system*.
    Thus seen from some other coordinate system, a given rotation will appear like a rotation and
    position-dependent translation.


    You can instantiate a CoordRotation by specifying the rotation in a variety of
    different ways.  Pick one of the following three:

    Parameters
    ----------
    tx,ty,tz : floats
        rotation angles around X, Y, Z axes, in that order, **in degrees**
    axis_angle : dict
        Dict of axis and angle. You can specify *either* this or tx as a parameter.
    rotmatrix : 3x3 ndarray
        Rotation matrix in the current coordinate system.


    Notes
    ------

    Specifying rotation is a tricky area full of tons of degeneracies.

    We choose the following convention here:  Rotations shall be represented
    by the so-called Tait-Bryan angles theta_X, theta_Y, theta_Z which represent
    rotation around the *fixed* axes X, Y, then Z of the coordinate system in that
    order. These are like the standard Euler angles, except that Euler angles are
    defined for axes that rotate along with the moving body. Here we are concerned
    instead with a fixed coordinate axis system and a rotation of some object with
    respect to that system.

    This class allows one to convert that rotation into the required theta_X,Y,Z for
    some other fixed coordinate axes.
   

    """

    def __init__(self, arg1, ty=None, tz=None, system=None, unit='deg', *value, **args):
        if system is None:
            system = CoordAxesGlobal()

        self.system = system

        #predefine some attributes here so we can set the docstrings properly for
        #parsing by sphinx autodoc. The actual values will be set below

        
        self.axis_angle = dict()
        """ Axis-angle representation of a rotation, stored as a dict containing keys 'axis' and 'angle' """

        self.angles = np.zeros(3)
        """ Euler angles of a matrix """


        if isinstance(arg1, dict):
        #if isinstance(arg1, dict) and ty is None and tz is None:
            # we are given an axis-angle representation, so convert that to a
            # rotation matrix.

            self.axis_angle = arg1


            # make sure the axis vector is in the  local coord system
            self.axis_angle['axis'] = self.axis_angle['axis'].toLocal(system)
           

            # infer rot matrix and euler angles from the axis-angle representation.
            self.rotation_matrix= self._matrix_from_axis_angle()
            self.angles = self._euler_angles_from_matrix()
        elif isinstance(arg1, np.ndarray) and arg1.shape == (3,3):
            # We are given a rotation matrix directly:
            self.rotation_matrix = np.matrix(arg1)
            self.axis_angle = self._axis_angle_from_matrix()
            self.angles = self._euler_angles_from_matrix()

        else:
            # compute matrix representation
            tx = arg1
            self.angles = [tx,ty,tz]

            self.rotation_matrix = RotZ(tz) * RotY(ty) * RotX(tx)
            self.axis_angle = self._axis_angle_from_matrix()


    def _matrix_from_axis_angle(self):
        """ Calculate a rotation matrix given the axis-angle representation. """
        # see e.g. http://en.wikipedia.org/wiki/Rotation_representation_(mathematics)#Euler_axis.2Fangle_.E2.86.94_quaternion
        axis = self.axis_angle['axis']
        try:
            angle = self.axis_angle['angle_radians']
        except:
            angle = self.axis_angle['angle_degrees'] *np.pi/180

        e1, e2, e3 = toTuple(axis)
       
        E = np.matrix([[0,  -e3,  e2],
                       [e3,   0, -e1],
                       [-e2, e1,  0 ]]) 
        I = np.matrix(np.identity(3))
        #rotmatrix = I * np.cos(angle) + (1- np.cos(angle)) * axis * axis.T + E * np.sin(angle)
        rotmatrix = I + E * np.sin(angle) + (1- np.cos(angle))*E *E
        return rotmatrix

    def _axis_angle_from_matrix(self):
        """ Calculate an axis-angle representation given a rotation matrix"""
        # compute axis-angle representation
        angle =  float(np.arccos( (self.rotation_matrix.trace() - 1 )/2.0  ))
        angle_d = angle/np.pi*180
        if angle == 0:
            axis = [1,0,0] # no rotation so axis doesn't really matter
        else:
            R = self.rotation_matrix
            axis = np.matrix([R[2,1] - R[1,2],
                             R[0,2]-R[2,0],
                            R[1,0]-R[0,1]] ) / (2*np.sin(angle))

        axisV=DisplacementVector(toTuple(axis), system=self.system)

        axis_angle = {'axis':axisV, 'angle_radians': angle, 'angle_degrees': angle_d}
        return axis_angle


    def _euler_angles_from_matrix(self):
        """ Compute the Euler angles given a rotation matrix"""
       
        # We decompose into the rotation angles around the X, Y, and Z axes, in order.
        # That yields the following matrix:
   
		#{
		# {Cos[ty] Cos[tz],     Cos[tz] Sin[tx] Sin[ty] - Cos[tx] Sin[tz],   Cos[tx] Cos[tz] Sin[ty] + Sin[tx] Sin[tz]},
		# {Cos[ty] Sin[tz],   Cos[tx] Cos[tz] + Sin[tx] Sin[ty] Sin[tz],    -Cos[tz] Sin[tx] +   Cos[tx] Sin[ty] Sin[tz]},
		# {-Sin[ty],            Cos[ty] Sin[tx],                            Cos[tx] Cos[ty]}
		#}

        R = self.rotation_matrix
        ty =  np.arcsin(-R[2,0]) 
        tx =  np.arctan2( R[2,1], R[2,2])
        tz =  np.arctan2( R[1,0], R[0,0])
        return np.array((tx, ty, tz))*180/np.pi

    def toLocal(self,system):
        """ convert to local coordinates in some other coordinate system
        
        Parameters
        ------------
        system : CoordAxes
            The coordinate system you wish to convert this rotation into

        Returns
        ----------
        newrot : CoordRotation
            A CoordRotation expressed in that system, equivalent to the same transformation as this rotation.


        """
        axis = self.axis_angle['axis']
        newlocalaxis= axis.toLocal(system)
        ax_angle = {'axis':axis,
                    'angle_degrees': self.axis_angle['angle_degrees'],
                    'angle_radians': self.axis_angle['angle_radians']}

        return CoordRotation(ax_angle, system=system)

    def toGlobal(self):
        """Convert to global coordinates 
        
        Returns
        --------
        globalrot : CoordRotation
            A CoordRotation expressed in the global frame, equivalent to this rotation
        """
        return self.toLocal(system=CoordAxesGlobal)

    # This is sort of an unpythonic hack to overload the repr and str differently to handle the two
    # desireable output formats. Fix later? Too bad you can't use keywords with repr or str...
    def __repr__(self):
        return '<CoordRotation by %f deg around %s>' % (self.axis_angle['angle_degrees'], repr(self.axis_angle['axis']))
    def __str__(self):
        return "<Rotation by  [%g, %g, %g] deg in system '%s'>" % (self.angles[0], self.angles[1], self.angles[2], self.system.name)

    def __mul__(self, vector):
        """ Return a new vector which is rotated by the desired amount.
        Does not change the original vector """

        try:
            localvector = vector.toLocal(self.system)
        except:
            localvector = vector # this catches the case where we multiply a CoordRotation times a regular ndarray, in which case we just assume they must be in the same system.
        newvector = self.rotation_matrix * localvector
        return CoordVector(newvector, system=self.system)

class CoordAxes(object):
    """

    Fundamental class for Coordinate Systems

    Parameters
    -----------
    name : string
        A descriptive name for this coordinate system
    origin :  CoordVector or 3-element ndarray or other iterable
        vector to origin of coords in global system.
    ux, uy, uz : CoordVector or 3-element ndarray or other iterable
        unit vectors in x,y,z directions, in global system.
        In most cases these will be orthonormal but that is not strictly enforced
    """

    def __init__(self,origin, ux,uy,uz, name=None):
        if name is None:
            name = "Axes at [%f,%f,%f]" % toTuple(origin)

        self.name=name
        # FIXME check the case where the inputs are already Coordvector objects?
        self._origin = CoordVector(origin)
        self._x = CoordVector(ux)
        self._y = CoordVector(uy)
        self._z = CoordVector(uz)
        self._basis = CoordVector(np.zeros((3,3)))
        self._basis[:,0] = self._x
        self._basis[:,1] = self._y
        self._basis[:,2] = self._z

        self._coordsys = CoordSys.CoordSys(toTuple(self._origin), toTuple(self._basis[:,0]), toTuple(self._basis[:,1]), toTuple(self._basis[:,2]))

        #self._coordsys = CoordSys.CoordSys(self._origin.toTuple(), self._x.toTuple(), self._y.toTuple(), self._z.toTuple())


    def __str__(self):
        return str(self._coordsys)

    def rotateAxesLocal(self, rotationmatrix):
        """ Return a rotated version of the coordinate axis, rotated around *its own origin*
        The origin of the returned CoordAxes coincides with the origin of this original CoordAxes.
        
        Parameters
        ----------
        rotationmatrix : matrix
            Description of the desired rotation to apply

        Returns
        -------
        newcoordsys : CoordSys
            New coordinate system that has been rotated as requested

        """
        newbasis = rotationmatrix * self._basis

        return CoordAxes(self._origin, newbasis[:,0], newbasis[:,1], newbasis[:,2], name='rotated '+self.name )

    def rotateAxesGlobal(self, rotationmatrix):
        """ Return a rotated version of the coordinate axis, rotated around *the global origin*
        The origin of the returned CoordAxes will differ from the origin of this original CoordAxes.
        
        Parameters
        ----------
        rotationmatrix : matrix
            Description of the desired rotation to apply

        Returns
        -------
        newcoordsys : CoordSys
            New coordinate system that has been rotated as requested


        """
        newbasis = rotationmatrix * self._basis
        neworigin = rotationmatrix * self._origin
        return CoordAxes(neworigin, newbasis[:,0], newbasis[:,1], newbasis[:,2], name='rotated+translated '+self.name)


    def draw3d(self, color='rgb', labels=None, labelpos=1.8, length=200, ax=None, **kwargs):
        """ Draw on screen a representation of the axes basis 
        
        Parameters
        -----------
        color : string, optional
            matplotlib color specification for the axes to be drawn. 
            Note: the special color string 'rgb' will cause the X,Y,Z axes to be
            drawn in red, green, and blue respectively. This is the default behavior.
        labels : iterable of strings, optional
            Descriptive names for the axes to be drawn on screen
        labelpos : float, optional
            offset for drawing the label text
        length : float, optional
            length of vectors to draw
        ax : matplotlib.Axes, optional
            Axes to draw into. If not set, the current axes will be used. 
        
        """
        if ax is None:
            ax = pl.gca()
        cen = np.array(self._origin).flatten()
        bx= np.array(self._basis[:,0]).flatten()
        by= np.array(self._basis[:,1]).flatten()
        bz= np.array(self._basis[:,2]).flatten()

        if color=='rgb': colors=['r','g','b']
        else: colors = [color]*3

        if isinstance(labels, str): labels = [labels]*3 # for one string, replicate 3x

        for i, b in enumerate([bx,by,bz]):
            arrowcoords = np.zeros((2,3))
            arrowcoords[0] = cen
            arrowcoords[1] = cen+b*length
            ax.plot(arrowcoords[:,0].flatten(),arrowcoords[:,1].flatten(),arrowcoords[:,2].flatten(),color=colors[i], zorder=300, **kwargs)

            if labels is not None:
                labelc = cen+b*length*labelpos
                ax.text(labelc[0], labelc[1], labelc[2], labels[i], color=colors[i],size='small', horizontalalignment='center',verticalalignment='center', zorder=301)


    def toLocal(self, vec):
        """ Convert a given vector into its local representation in this coordinate system

        Parameters
        -----------
        vec : CoordVector
            Vector to be converted

        Returns
        ---------
        localvec : CoordVector
            A copy of the input vector, now referenced to this coordinate system.
        """

        return CoordVector( self._coordsys.toLocal(vec), system=self)
    def toGlobal(self, vec):
        """ Convert a given vector into its global representation 

        Parameters
        -----------
        vec : CoordVector
            Vector to be converted

        Returns
        ---------
        localvec : CoordVector
            A copy of the input vector, now referenced to the global coordinate system
        """


        return CoordVector(self._coordsys.toGlobal(vec), system = CoordAxesGlobal())
    def origin(self):
        """ Return the origin of this coordinate system, in global coordinates 
        
        Returns
        --------
        originvector : CoordVector
            A vector representing the global origin of this coordinate system

        """
        if isinstance(self._origin, CoordVector):
            return CoordVector(self._origin.toGlobal(), system = CoordAxesGlobal())
        else:  # it's just a list or tuple
            return CoordVector(self._origin, system = CoordAxesGlobal())



    def __repr__(self):
        return "<CoordAxes: "+self.name+">"

class CoordAxesGlobal(CoordAxes):
    """Convenience class to return the global coordinate system.

    This is a singleton class - i.e. there is only one unique global instance.

    Parameters
    ----------
    none

    See Also
    ---------
    CoordAxes : Base coordinate systems class


    Examples
    ----------

    Obtain a handle to the global coordiante system:

    >>> gl = CoordAxesGlobal()


    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CoordAxesGlobal, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):

        self.name='_global_'
        # FIXME check the case where the inputs are already Coordvector objects?
        self._origin = [0,0,0]
        self._basis = np.matrix(np.zeros((3,3)))
        self._basis[0,0] = 1
        self._basis[1,1] = 1
        self._basis[2,2] = 1
        toTuple = lambda thing: tuple(np.array(thing).flatten())


        self._coordsys = CoordSys.CoordSys(toTuple(self._origin), toTuple(self._basis[:,0]), toTuple(self._basis[:,1]), toTuple(self._basis[:,2]))

#---------------

class test_CoordRotations(unittest.TestCase):
    def test_axis_angle(self):
        """ Assert that we can convert back from axis-angle to rotation matrix properly
        """

        rots = [ [90,0,0], [0,90,0], [0,0,90], [0,0,-90], [90,90,0]]

        for rot in rots:
            CR = CoordRotation(*rot)

            R1 = toTuple(CR.rotation_matrix)

            rotmatrix = CR._matrix_from_axis_angle()
            R2 = toTuple(rotmatrix)

            for i in range(len(R1)):
                self.assertAlmostEqual( R1[i], R2[i])

    def test_creation(self):
            CR1 = CoordRotation(90,0,0)
            CR2 = CoordRotation(RotX(90))
            aa = {'axis': CoordVector([1,0,0]), 'angle_degrees':90}
            CR3 = CoordRotation(aa)


            for ix in range(3):
                for iy in range(3):
                    self.assertAlmostEqual( CR1.rotation_matrix[iy,ix], CR2.rotation_matrix[iy,ix])
                    self.assertAlmostEqual( CR1.rotation_matrix[iy,ix], CR3.rotation_matrix[iy,ix])

    def test_rotating_vectors(self):
        # X rotation by 90 deg should swap Y and Z
        rot = [90,0,0]
        CR = CoordRotation(*rot)

        vec  = CoordVector([1,0,0])
        vec2  = CR*vec

        for i in range(3):
            self.assertAlmostEqual( vec[0], vec2[0])
            self.assertAlmostEqual( vec[1], vec2[2])
            self.assertAlmostEqual( vec[2], -vec2[1])

        #rotation should not change vector lengths
        rots = [ [90,0,0], [0,90,0], [0,0,90], [0,0,-90], [90,90,0]]
        vecs = [ [1,0,0], [0,1,0], [0,0,1], [0,0,-1], [1,1,0]]

        for rot in rots:
            for v in vecs:
                V1 = CoordVector(v)
                V2 = CoordRotation(*rot) * V1
                self.assertAlmostEqual( V1.mag(), V2.mag())


    def test_euler_from_matrix(self):
        rots = [ [90,0,0], [0,90,0], [0,0,90], [0,0,-90], [90,90,0], [45, 12, 0], [-20,0,30], [30,30,50],[-90,90,0],
                [90,90,90] ]
        for rot in rots:
            CR = CoordRotation(*rot)
            res = CR._euler_angles_from_matrix()
            for i in range(3):
                self.assertAlmostEqual(rot[i], res[i])


if __name__ == '__main__':
   
    pass
