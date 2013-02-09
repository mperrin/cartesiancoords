import unittest

from .cartesian import *   

class test_CoordSys(unittest.TestCase):
    def test_global(self):
        g = CoordAxesGlobal()

        vecs = [ [1,0,0], [0,1,0], [0,0,1], [0,0,-1], [1,1,0]]

        for v in vecs:

            vec = CoordVector(v)
            t1 = vec.toGlobal()
            t2 = vec.toLocal(g)

            for i in range(3):
                self.assertEqual( v[i], vec[i])
                self.assertEqual( v[i], t1[i])
                self.assertEqual( v[i], t2[i])


    def test_translate(self):
        dx,dy, dz = 100, 50, -3.5

        cs = CoordAxes([dx,dy,dz], [1,0,0],[0,1,0],[0,0,1], name='test' )
        vecs = [ [1,0,0], [0,1,0], [0,0,1], [0,0,-1], [1,1,0]]

        for v in vecs:
            vec = CoordVector(v)
            vl = vec.toLocal(cs)
            self.assertEqual( v[0]-dx, vl[0])
            self.assertEqual( v[1]-dy, vl[1])
            self.assertEqual( v[2]-dz, vl[2])

    def test_rotated_coords(self):
        # 90 degree rotation
        cs = CoordAxes( [0,0,0], [0,1,0],[-1,0,0],[0,0,1], name='test' )
        vecs = [ [1,0,0], [0,1,0], [0,0,1], [0,0,-1], [1,1,0]]

        for v in vecs:
            vec = CoordVector(v)
            vl = vec.toLocal(cs)
            self.assertEqual( v[1], vl[0])
            self.assertEqual( -v[0], vl[1])
            self.assertEqual( v[2], vl[2])


    def test_displacement(self):

        for system in (CoordAxesGlobal(), CoordAxes( [1,0,0], [0,1,0],[0,0,1],[1,0,0]), CoordAxes( [1,0,-370], [0,0.5,0],[0,0,1],[1,1,0])):

            V1 = CoordVector([10,0,0], system=system)
            V2 = CoordVector([0,10,0], system=system)
            V3 = CoordVector([0,0,10], system=system)

            disp12 = V1- V2
            disp23 = V2- V3
            self.assertTrue( isinstance(disp12, DisplacementVector))
            self.assertTrue( isinstance(disp23, DisplacementVector))

            disp13 = disp12+disp23

            newV1 = V3+disp13
            newV3 = V1-disp13

            far = V1 - 10*disp12 + (-10)*disp23
            newV3b = far +9*(disp12+disp23)

            self.assertTrue( isinstance(disp13, DisplacementVector))
            self.assertTrue( isinstance(newV1, CoordVector))
            self.assertTrue( isinstance(newV3, CoordVector))
            for i in range(3):
                self.assertEqual(disp12[i], V1[i]-V2[i])
                self.assertEqual(V1[i], newV1[i])
                self.assertEqual(V3[i], newV3[i])
                self.assertEqual(V3[i], newV3b[i])


    def test_displacement2(self):
        " displacement math across multiple coordinate systems"

        other = CoordAxes( [-1,0,0], [0,1,0],[0,0,1],[1,0,0])
        for CA in (CoordAxesGlobal(), CoordAxes( [1,0,0], [0,1,0],[0,0,1],[1,0,0]), CoordAxes( [1,0,-370], [0,0.5,0],[0,0,1],[1,1,0])):

            vec1 = CoordVector([10,0,0])
            vec2 = CoordVector([10,0,0], system=CA)
            vec2_g = vec2.toGlobal()
            disp12 = vec2-vec1

            #1st subscript letter indicates which version of the displacement was used
            disp12_g = disp12.toGlobal()
            disp12_l = disp12.toLocal(CA)
            disp12_o = disp12.toLocal(other)

            new2_g = vec1 + disp12_g
            new2_l = vec1 + disp12_l
            new2_o = vec1 + disp12_o

            # 2nd subscript gives the coordinate system for the resulting point.
            new2_g_g = new2_g.toGlobal()
            new2_g_l = new2_g.toLocal(CA)
            new2_l_g = new2_l.toGlobal()
            new2_l_l = new2_l.toLocal(CA)
            new2_o_g = new2_o.toGlobal()
            new2_o_l = new2_o.toLocal(CA)

            for i in range(3):
                self.assertEqual(new2_g_l[i], new2_l_l[i])
                self.assertEqual(new2_g_g[i], new2_l_g[i])
                self.assertEqual(new2_g_g[i], vec2_g[i])
                self.assertEqual(new2_g_l[i], vec2[i])
                self.assertEqual(new2_g_l[i], new2_o_l[i])
                self.assertEqual(new2_g_g[i], new2_o_g[i])





def test_plot():
    try:
        pl.clf()
    except:
        pl.clf()
    g = CoordAxesGlobal()

    dx,dy, dz = 100, 50, 200
    cs = CoordAxes([dx,dy,dz], [1,0,0],[0,-1,0],[0,1,1])

    pl.gcf().add_subplot(111, projection='3d', aspect='equal',ymargin=0, xmargin=0)
    g.draw3d(labels='global', color='blue')
    cs.draw3d(color='red', labels='local')


    for coords in ([1000,100,0], [300,1000,0], [300,300,300]):
        gvec = CoordVector(coords)
        lvec = gvec.toLocal(cs)

        gvec.draw3d()
        lvec.draw3d(linestyle='--')

    pl.draw()

def run_tests():
    unittest.main()


if __name__ == '__main__':
    run_tests() 
