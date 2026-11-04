"""Reference arithmetic tests using Python standard library only."""
import unittest, math

def conv(a,k):
 h,w=len(a),len(a[0]);out=[[0.]*w for _ in range(h)]
 for y in range(h):
  for x in range(w):
   for dy in range(3):
    for dx in range(3):
     iy=y+dy-1;ix=x+dx-1
     if 0<=iy<h and 0<=ix<w: out[y][x]+=a[iy][ix]*k[dy][dx]
 return out

def bn(a,gamma=1,beta=0,eps=1e-5):
 v=[x for row in a for x in row];mean=sum(v)/len(v);var=sum((x-mean)**2 for x in v)/len(v)
 return [[gamma*(x-mean)/math.sqrt(var+eps)+beta for x in row] for row in a]

def silu(x):
 q=math.exp(-abs(x));s=1/(1+q) if x>=0 else q/(1+q)
 return x*s

class Tests(unittest.TestCase):
 def test_identity(self):
  a=[[float(i+j*4) for i in range(4)] for j in range(3)]
  self.assertEqual(conv(a,[[0,0,0],[0,1,0],[0,0,0]]),a)
 def test_blur(self):
  r=conv([[1.]*3 for _ in range(3)],[[1/9]*3 for _ in range(3)])
  self.assertAlmostEqual(r[1][1],1.);self.assertAlmostEqual(r[0][0],4/9)
 def test_edge(self):
  r=conv([[1.]*5 for _ in range(5)],[[-1,0,1],[-2,0,2],[-1,0,1]])
  self.assertEqual(r[2][2],0.)
 def test_shape(self):
  x=conv([[0.]*16 for _ in range(12)],[[1.]*3 for _ in range(3)])
  self.assertEqual((len(x),len(x[0])),(12,16))
 def test_bn_mean(self):
  v=bn([[0,1,2,3]])[0];self.assertAlmostEqual(sum(v),0)
 def test_bn_beta(self):self.assertEqual(bn([[1.,2.,3.]],0,2),[[2.,2.,2.]])
 def test_bn_constant(self):self.assertEqual(bn([[7.,7.]],1,-.2),[[-.2,-.2]])
 def test_silu_zero(self):self.assertEqual(silu(0),0.)
 def test_silu_known(self):
  self.assertAlmostEqual(silu(1),.73105857863,places=10)
  self.assertAlmostEqual(silu(-1),-.26894142137,places=10)
 def test_csp_concept(self):
  x=[[[float(c) for c in range(4)] for _ in range(16)] for _ in range(12)]
  bank=[[[x[y][col][ch] for col in range(16)] for y in range(12)] for ch in range(4)]
  blur=[[1/9]*3 for _ in range(3)];edge=[[-1,0,1],[-2,0,2],[-1,0,1]]
  combined=[bank[0],bank[1],conv(conv(bank[2],blur),edge),conv(conv(bank[3],edge),blur)]
  self.assertEqual((len(combined),len(combined[0]),len(combined[0][0])),(4,12,16))
  self.assertEqual(combined[:2],bank[:2])

if __name__=="__main__":unittest.main()
