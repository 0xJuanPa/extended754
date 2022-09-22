import random



"""

TEST FILE 
WOULD BE GOOD IF RUN IT UNDER A DEBUGGER TO SEE REPRESENTATIONS AND CALCS

"""
# importamos la clase principal de la biblioteca
from extended754 import Extended754
# aritmetica binaria de precision doble como en IEE754
# usaremos representacion precisa de 17 digitos que en python oscila de 15 a 17 digitos
base2Double = Extended754(base=2, exponent=11, mantissa=53, repr_len=17)  # 53 explicito porque no tiene hidden bit
n1 = base2Double("0,1", 10)  # creamos el efloat 0,1 entrado en base 10
n2 = base2Double("0,2", 10)  # creamos el efloat 0,2 entrado en base 10
# sumamos los efloats
n3 = n1 + n2
# hacemos la misma operacion con el float de python para comparar
pn3 = f"{0.1 + 0.2:.{base2Double.repr_len}f}"
# comparamos
assert str(n3) == pn3

# aritmetica binaria de precision simple como en IEE754
base2Simple = Extended754(2, 8, 24)  # 24 explicito porque no tiene hidden bit

numx2 = base2Simple("3.999999761589", 10)  # overflow de redondeo
numx3 = base2Simple("0.00000000000", 10)
numx5 = base2Simple("0,81", 10)


# ejemplos de prueba selectos
for i, j in [("0.05", "3"), ("100", "100"), ("100", "0.5"), ("1", "3"), ("0.5", "1.5"), ("1.5", "0.5"), ("0.5", "0.5"),
             ("1", "0.5"), ("0.5", "1")]:
    num1 = base2Double(i, 10)
    num2 = base2Double(j, 10)
    num1f = float(i)
    num2f = float(j)
    num3 = num1 + num2
    pnum3 = f"{num1f + num2f:.{base2Double.repr_len}f}"
    assert str(num3) == pnum3
    num4 = num1 - num2
    pnum4 = f"{num1f - num2f:.{base2Double.repr_len}f}"
    assert str(num4) == pnum4
    num5 = num1 * num2
    pnum5 = f"{num1f * num2f:.{base2Double.repr_len}f}"
    assert str(num5) == pnum5
    num6 = num1 / num2
    pnum6 = f"{num1f / num2f:.{base2Double.repr_len}f}"
    assert str(num6) == pnum6


# para comprobar puede subst op con la operacion entre numeros floats 64 bit base2 de python e imprimir 25 digitos
# como la implementacion es generica debe funcionar para las demas bases
# print(f"{op:.{base2Double.repr_len}f}")


# for spec case
f1 = 3*10**16 # test it with exp 15 and exp 16 where loss of significance due to high diferences betwen exp
f2 = 0.5
op = lambda i0, i1: i0 + i1
x = base2Double(str(f1), 10)
y = base2Double(str(f2), 10)
rsf = f"{op(f1,f2):.{base2Double.repr_len}f}"
res = op(x,y)
assert str(res) == rsf

# creamos una aritmetica de base 3 con representacion en base 3
base3Arithm = Extended754(base=3,exponent=6,mantissa=12,repr_base=3)
one = base3Arithm("1",10) # creamos efloat
three = base3Arithm("3",10) # creamos efloat
# este número no es periodico en base 3
res = one/three
b3 = str(res) # observamos el número en base 3 ->
# después de cambiar la base en la q queremos observar el número 0.1 sera 0.3333...
base3Arithm.repr_base=10
b10 = str(res) # observamos el número en base 10 ->


a = base2Double(26.25240569327435,10)
b = base2Double(47.07254106288465,10)
res = a+b
repr(res)


# super tester with double prec binary arithm against python
# reprlen to 10 to avoid dtoa precision "speculation" because we ar comparing ourselves with python
base2Double.repr_len = 10
# iterate over different operations implemented
for op in [lambda i0, i1: i0 + i1,lambda i0, i1: i0 * i1,lambda i0, i1: i0 - i1,lambda i0, i1: i0 / i1]:
    # generate some random numbers to test each ops
    for i in range(1000):
        xf = random.uniform(-100, 100) # change numeric ranges here
        yf = random.uniform(-100, 100) # change numeric ranges here
        x = base2Double(str(xf), 10)
        y = base2Double(str(yf), 10)
        res = op(x, y) # perform the operation betwen efloats
        rsf = f"{op(xf, yf):.{base2Double.repr_len}f}" # perform the operation in python floats
        try:
            assert str(res) == rsf # compare string representation
        except:
            print(f"Error in\ne754: {res}\n py  : {rsf}")

