import functools
import math
import re
from typing import Union


def get_number(digits: [int], base: int) -> int:
    """
    :param digits: mantissa
    :param base: base
    :return: python integer
    Converts a mantissa in the specified base to python integer
    Generalization of binary conversion algorithm
    """
    x = functools.reduce(lambda n, d: n * base + d, digits) if digits else 0
    return x


def int_arr_in_base(x: int, base: int) -> [int]:
    """
    :param x: python integer
    :param base: mantissa base
    :return: mantissa in specified base
    Converts a python integer to mantissa in specified base
    Generalization of binary conversion algorithm
    """
    digits = []
    while x > 0:
        digits = [x % base] + digits
        x = x // base
    return digits if len(digits) else [0]


def frac_arr_converter(digits: [int], in_base: int, out_base: int, frac_places: int, normalized=False) -> [int]:
    """
    :param digits:
    :param in_base:
    :param out_base:
    :param frac_places:
    :param normalized:
    :return:

    Converts a fractionary array of digits in one base to another array of digits in other base
    Generalization of binary conversion algorithm
    """
    int_calc = get_number(digits, in_base)
    comma_place = in_base ** len(digits)
    frac = []
    normalized_count = 0
    while len(frac) < frac_places:
        int_calc *= out_base
        int_part = int_calc // comma_place
        if (normalized and int_part != 0) or not normalized or len(frac):
            frac += int_arr_in_base(int_part, out_base)
        else:
            normalized_count += 1
        int_calc %= comma_place
        if int_calc == 0:
            break
    return frac, normalized_count


def nan_protect(func):
    '''
    :param func: Function to Protect

    This Decorator Protects operations for being executed with NaN values and returns NaN
    '''

    def wrapper(self: "efloat", other: "efloat"):
        if self.isNaN or other.isNaN:
            return self.arithm_ref.NaN.copy()
        return func(self, other)

    return wrapper


class efloat:
    def __init__(self, arithm: "Extended754", sign: int, exponent: int, mantissa: [int]):
        self.arithm_ref: "Extended754" = arithm
        self.sign: int = sign
        self.exponent: int = exponent
        self.mantissa = mantissa

    @property
    def isNaN(self):
        """
        :return: if the current efloat is NaN
        """
        return abs(self.exponent) == self.arithm_ref.exp_bias + 1 \
               and get_number(self.mantissa, self.arithm_ref.base) > 0

    @property
    def isZero(self):
        """
        :return: if the current efloat is zero
        """
        return abs(self.exponent) == 0 \
               and get_number(self.mantissa, self.arithm_ref.base) == 0

    @property
    def isINF(self):
        """
        :return: if the current efloat is infinity
        """
        return abs(self.exponent) == self.arithm_ref.exp_bias + 1 and get_number(self.mantissa,
                                                                                 self.arithm_ref.base) == 0

    def copy(self) -> "efloat":
        """
        :return: A shallow copy of current efloat
        """
        return efloat(self.arithm_ref, self.sign, self.exponent, self.mantissa[:])

    @nan_protect
    def __add__(self, other: "efloat"):
        if self.isINF:
            return self.arithm_ref.pInf.copy() if self.sign == 1 else self.arithm_ref.nInf.copy()
        if other.isINF:
            return self.arithm_ref.pInf.copy() if other.sign == 1 else self.arithm_ref.nInf.copy()

        # get the numbers with max and min exponent in respective places
        max_exp = self if self.exponent >= other.exponent else other
        min_exp = other if other.exponent <= self.exponent else self
        # calculate the offset of exponents
        delta = min(abs(max_exp.exponent - min_exp.exponent), self.arithm_ref.mantissa_len)

        # shift the mantissa of the min exponent number
        shifted_mantissa = ([0] * delta) + min_exp.mantissa[:self.arithm_ref.mantissa_len - delta]

        # convert the numbers to python int
        max_n = get_number(max_exp.mantissa, self.arithm_ref.base) * max_exp.sign
        min_n = get_number(shifted_mantissa, self.arithm_ref.base) * min_exp.sign

        # perform the operation
        result = max_n + min_n
        # convert back to mantissa representation
        arr_result = int_arr_in_base(abs(result), self.arithm_ref.base)

        # calculate sign and exponent
        exponent = max_exp.exponent
        sign = -1 if result < 0 else 1

        # the result of operations with integers already normalizes the array but need to adjust exponents
        # as the mantissa can lose or gain places
        len_result = len(arr_result)
        if len_result > self.arithm_ref.mantissa_len:
            exponent += 1
        elif len_result < self.arithm_ref.mantissa_len:
            exp_loss = (self.arithm_ref.mantissa_len - len_result)
            arr_result = arr_result + ([0] * exp_loss)
            exponent -= exp_loss

        # round the result
        arr_result = self.arithm_ref.eround(sign, arr_result, self.arithm_ref.mantissa_len)
        # renormalize in case
        exponent, arr_result = self.arithm_ref.normalize(exponent, arr_result)
        # construct the efloat with the calculated values
        res = efloat(self.arithm_ref, sign, exponent, arr_result)
        return res

    def __neg__(self):
        # get a shallow copy of self
        self_copy: "efloat" = self.copy()
        # change sign
        self_copy.sign *= -1
        return self_copy

    @nan_protect
    def __sub__(self, other: "efloat"):
        # negate other
        other_negated = -other
        # perform efloat sum
        res = self + other_negated
        return res

    @nan_protect
    def __mul__(self, other: "efloat"):
        if self.isINF and other.isZero or self.isZero and other.isINF:
            return self.arithm_ref.NaN.copy()

        if self.isINF and not other.isZero or not self.isZero and other.isINF:
            return self.arithm_ref.pInf.copy() if self.sign * other.sign == 1 else self.arithm_ref.nInf.copy()

        # convert the numbers to python int
        x = get_number(self.mantissa, self.arithm_ref.base) * self.sign
        y = get_number(other.mantissa, self.arithm_ref.base) * other.sign
        # perform the operation
        result = x * y
        # convert back to mantissa representation
        arr_result = int_arr_in_base(abs(result), self.arithm_ref.base)

        # calculate sign and exponent
        sign = -1 if result < 0 else 1
        exponent = self.exponent + other.exponent

        # the result of operations with integers already normalizes the array but need to adjust exponents
        # as the mantissa can lose or gain places
        len_result = len(arr_result)
        if len_result == self.arithm_ref.mantissa_len * 2:
            exponent += 1

        # apply the arithmetic defined rounding
        arr_result = self.arithm_ref.eround(sign, arr_result, self.arithm_ref.mantissa_len)
        # renormalize
        exponent, arr_result = self.arithm_ref.normalize(exponent, arr_result)
        # create the efloat instance with the calculated values
        res = efloat(self.arithm_ref, sign, exponent, arr_result)
        return res

    @nan_protect
    def __truediv__(self, other: "efloat"):
        if (self.isZero and other.isZero) or (self.isINF and other.isINF):
            return self.arithm_ref.NaN.copy()

        if other.isINF:
            return self.arithm_ref.pZero.copy() if self.sign * other.sign == 1 else self.arithm_ref.nZero.copy()

        if other.isZero:
            return self.arithm_ref.pInf.copy() if self.sign * other.sign == 1 else self.arithm_ref.nInf.copy()

        # convert other to string then to python float
        y = float(other.convert_str(10).replace(",", "."))  # heuristic estimation
        initial_guess = 1 / y  # claculate the initial gess using base2 error but good estimation
        # convert initial guess to str and then parse it with the same arithmetic
        initial_guess_str = f"{initial_guess:.{self.arithm_ref.mantissa_len}f}"
        initial_guess_ef = self.arithm_ref(initial_guess_str, 10)
        # Newton-Raphson whithout lookup tables just using as approximation the reciprocal
        # calculated in base 2 for initial guess
        two = self.arithm_ref("2", 10)  # this is the number 2 but repr in the arithm for the calcs
        y_0 = initial_guess_ef
        iterations = 0
        # iterate until it converges
        while True:
            iterations += 1
            # NR step
            y_n = y_0 * (two - other * y_0)
            # ask if NAN because NaN != NaN other
            if not y_n.isNaN and y_0 != y_n and iterations < self.arithm_ref.nr_it: # oscillations can appear
                y_0 = y_n
            else:
                break
        # perform the operation, division is the most costly and harder operation, this implementation
        # relies on the sum and multiplication
        res = self * y_n
        return res

    def __gt__(self, other: "efloat"):
        if self.isNaN or other.isNaN:
            return False
        if self.isZero and other.isZero:
            return False
        if self.sign > other.sign:
            return True
        if self.sign < other.sign:
            return False
        if self.exponent > other.exponent:
            return True
        if self.exponent < other.exponent:
            return False
        return get_number(self.mantissa, self.arithm_ref.base) > get_number(other.mantissa, self.arithm_ref.base)

    def __eq__(self, other: "efloat"):
        if self.isNaN or other.isNaN:
            return False
        if self.isZero and other.isZero:
            return True
        res = self.sign == other.sign and self.exponent == other.exponent \
               and get_number(self.mantissa, self.arithm_ref.base) == get_number(other.mantissa, self.arithm_ref.base)
        return res

    def __ge__(self, other: "efloat"):
        return self > other or self == other

    def __lt__(self, other):
        return other > self

    def __le__(self, other):
        return other > self or self == other

    def convert_str(self, repr_base: int, sep: str = ".") -> str:
        """
        :param repr_base: The base in which want to represent the number on the str
        :param sep:  separator for decimal places char
        :return:  string representation of current number

        This is like dtoa or equivalent fucntion in gnu. represents the current efloat to string
        """
        sign = "-" if self.sign == -1 else ""
        if self.isINF:
            return f"{sign}Inf"

        if self.isNaN:
            return "NaN"

        int_part_num = 0
        int_part = [0]

        if self.exponent < 0:
            frac_mantissa = ([0] * abs(self.exponent + 1)) + self.mantissa  # -1 because "0," we dont want the 0
        else:
            # guarantee index in bounds
            e = min(self.exponent + 1, self.arithm_ref.mantissa_len)
            # this is the completion for the mantissa len
            extra = [0] * (((self.exponent + 1) - self.arithm_ref.mantissa_len)
                           if self.exponent > self.arithm_ref.mantissa_len else 0)
            int_part_num = get_number(self.mantissa[:e] + extra, self.arithm_ref.base)
            int_part = int_arr_in_base(int_part_num, repr_base)
            frac_mantissa = self.mantissa[e:]

        # get if will use fixed precision representation or estimated one
        if self.arithm_ref.repr_len is None:
            precision = (self.arithm_ref.estimated_repr_len - len(int_part) if int_part_num > 0
                    else self.arithm_ref.estimated_repr_len)
        else:
            precision = self.arithm_ref.repr_len

        extra_prec = precision + self.arithm_ref.round_extra_places
        frac_part, _ = frac_arr_converter(frac_mantissa, self.arithm_ref.base, repr_base, extra_prec, False)

        starts0 = frac_part[0] == 0  # this is a trick to avoid rounding int number conversion cut zeroes at beginning
        if starts0:
            frac_part = [1] + frac_part
            precision += 1

        frac_part = self.arithm_ref.eround(1, frac_part, precision, repr_base)
        if len(frac_part) > precision:  # sum extra one to the int part if increased as round
            frac_part.pop(0)
            int_part_num += 1
            int_part = int_arr_in_base(int_part_num, repr_base)

        if starts0:  # reset of the trick
            frac_part.pop(0)
            precision -= 1

        # represent the int part
        intstr = "".join(map(lambda x: self.arithm_ref.char_map[x], int_part))

        # pad or unpad with 0 for fixed precision reresentation
        if self.arithm_ref.repr_len is None:
            while len(frac_part) > 1 and frac_part[-1] == 0:
                frac_part.pop()
        else:
            frac_part += [0] * (precision - len(frac_part))

        # represent the frac part
        frac_str = "".join(map(lambda x: self.arithm_ref.char_map[x], frac_part))
        # join all
        res = f"{sign}{intstr}{sep}{frac_str}"
        return res

    def __repr__(self):
        return self.convert_str(self.arithm_ref.repr_base)


class Extended754:
    def __init__(self, base: int, exponent: int, mantissa: int, round_nearest: bool = True, repr_base: int = 10,
                 char_map: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", repr_len: Union[int , None] = None):
        '''
        :param base: Arithmetic Base
        :param exponent: Maximum amount of exponent places in defined base
        :param mantissa: Maximum amount of mantissa places in defined base
        :param round_nearest: Round To Nearest Ties Even Activated or will use Truncate instead
        :param repr_base: Representation Base used to convert the instances to string
        :param char_map: Character Map for representing the numbers on a string
        :param repr_len: Sets fixed representation len

        Creates an arithmetic with the defined parameters.
        '''

        if base < 2:
            raise Exception("Invalid Base")

        if repr_base < 2:
            raise Exception("Invalid Repr Base")

        self.repr_base = repr_base
        self.repr_len = repr_len
        self.estimated_repr_len = math.floor((mantissa - 1) * math.log(base, repr_base))
        self.nr_it = 10

        """
        the length of the representation is hard to implement accurately it varies between langs it commonly relies in 
        gnu dtoa function https://stackoverflow.com/questions/3173056/why-does-dtoa-c-contain-so-much-code
        
        made an acceptable dtoa here
        
        for example in double precision it seems in all langs to have 12 decimal places of precision after that it
        changes depending the lang and implementation
                        
        C# code
        double a = 26.25240569327435;
        double b = 47.07254106288465;
        var res = $"{a+b:F25}";
        
        C# out  : 73,3249467561590000000000000
        Py out  : 73.3249467561590080322275753
        e754 out: 73.3249467561589938213728601
                                ^   
        so it appears to be rounding there 
        im rounding at the end
        
        thats why (53-1)*log(2)=15 digits
        
        so will use this formula to get the max precision digits of the number 
        mathflor(mantissa_digits_count * log_base_repr(arithm_base))
        
        the number is fine, the method to convert it to represent it to string is just not very advanced
        https://www.exploringbinary.com/quick-and-dirty-floating-point-to-decimal-conversion/
        
        """

        self.round_nearest = round_nearest
        self.char_map = char_map
        self.base = int(base)
        if self.base > len(char_map):
            raise Exception("Need to supply a char_map for this base")
        self.round_extra_places = 4
        self.exponent_len = int(exponent)
        self.mantissa_len = int(mantissa)
        self.exp_bias = self.base ** (self.exponent_len - 1) - 1
        # create instances for special values
        self.NaN: "efloat" = efloat(self, 1, self.exp_bias + 1, [1] * self.mantissa_len)
        self.pInf: "efloat" = efloat(self, 1, self.exp_bias + 1, [0] * self.mantissa_len)
        self.nInf: "efloat" = efloat(self, -1, self.exp_bias + 1, [0] * self.mantissa_len)
        self.pZero: "efloat" = efloat(self, 1, 0, [0] * self.mantissa_len)
        self.nZero: "efloat" = efloat(self, -1, 0, [0] * self.mantissa_len)

    def __call__(self, number: Union[str, float], input_base: int = 0):
        """
        :param number: fractional number in input_base supporting . and , as decimal separator
        :param input_base: input base num of fractional number , if not supplied or eq 0 the base of arithmetic will be used
        :return: Instance of efloat compatible with this arithmetic

        Parses an input string or number in indicated base to current arithmetic efloat instance
        """
        input_base = int(input_base)
        if input_base == 0:
            input_base = self.base

        if input_base > len(self.char_map):
            raise Exception("Supplied base not covered in arithmetic charmap")

        valid_chars = self.char_map[:input_base]
        number = str(number)
        num = re.match(f"^(?P<sign>[-+])?(?P<int_part>[{valid_chars}]*)(?:[,.](?P<frac_part>[{valid_chars}]*))?$",
                       number)
        if num is None:
            raise Exception("Invalid number for this base and charmap")

        sign = -1 if num["sign"] == "-" else 1

        int_part = []
        if num["int_part"]:
            i_arr = [self.char_map.index(n) for n in num["int_part"]]
            i_num = get_number(i_arr, input_base)
            int_part = int_arr_in_base(i_num, self.base)

        if int_part[0] == 0:
            int_part.pop()
        len_int_part = len(int_part)
        exponent = len_int_part - 1

        frac_part = []
        if num["frac_part"]:
            f_arr = [self.char_map.index(n) for n in num["frac_part"]]
            frac_part, normf = frac_arr_converter(f_arr, input_base, self.base,
                                                  self.mantissa_len - len_int_part + self.round_extra_places,
                                                  exponent < 0)
            exponent -= normf

        len_frac_part = len(frac_part)

        if not len_frac_part and not len_int_part:
            exponent = 0
        mantissa = int_part + frac_part + ([0] * (self.mantissa_len - (len_int_part + len_frac_part)))
        mantissa = self.eround(sign, mantissa, self.mantissa_len)
        exponent, mantissa = self.normalize(exponent, mantissa)
        res = efloat(self, sign, exponent, mantissa)
        return res

    def normalize(self, exponent, mantissa):
        """
        :param exponent:
        :param mantissa:
        :return: exponent, mantissa
        Warning - This function modify the mantissa array

        Normalizes the mantissa adjusting the exponent
        """
        while len(mantissa) and mantissa[0] == 0:
            mantissa.pop()
            exponent += 1
        if len(mantissa) > self.mantissa_len:
            mantissa.pop(-1)
            exponent += 1
        return exponent, mantissa

    def eround(self, sign: int, mantissa, max_places: int, base: int = None):
        """
        :param sign:
        :param mantissa:
        :param max_places:
        :param base:
        :return: rounded mantissa

        Rounds the mantissa to the max places using the aritmethic defined behavior
        """
        len_round_info = len(mantissa) - max_places
        if len_round_info <= 0:
            return mantissa

        base = base or self.base
        mantissa_res = mantissa[:max_places]
        # round strategy
        if self.round_nearest:
            round_info = mantissa[max_places:]
            num = get_number(mantissa_res, base)  # as result of this function always will get normalized numbers
            if round_info[0] >= base // 2 \
                    or (round_info[0] == base // 2 and (get_number(round_info[1:], base) > 0 or num & 1)):
                num += 1
                mantissa_res = int_arr_in_base(num, base)
            return mantissa_res
        else:  # truncate
            return mantissa_res
