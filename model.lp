\* Source Pyomo model name=unknown *\

min 
x1:
+1 x2
+1 x3

s.t.

c_u_x4_:
+1 x2
+1 x3
<= 1

bounds
   0 <= x2 <= +inf
   0 <= x3 <= +inf
end
