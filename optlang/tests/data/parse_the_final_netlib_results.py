"""A parser for Thorsten Koch's final netlib results.model.lp

http://www.zib.de/koch/perplex/data/netlib/txt/

@article{Koch:2004:FNR:2308906.2309292,
 author = {Koch, Thorsten},
 title = {The Final NETLIB-LP Results},
 journal = {Oper. Res. Lett.},
 issue_date = {March, 2004},
 volume = {32},
 number = {2},
 month = mar,
 year = {2004},
 issn = {0167-6377},
 pages = {138--142},
 numpages = {5},
 url = {http://dx.doi.org/10.1016/S0167-6377(03)00094-4},
 doi = {10.1016/S0167-6377(03)00094-4},
 acmid = {2309292},
 publisher = {Elsevier Science Publishers B. V.},
 address = {Amsterdam, The Netherlands, The Netherlands},
 keywords = {Linear-programming, NETLIB, Rational-arithmetic},
}
"""

import glob
import gzip
import os
import pickle
import re
from fractions import Fraction

import six

OBJ_REGEX = re.compile('\* Objvalue : -?\d+/\d+')

the_final_netlib_results = dict()

for path in glob.glob("netlib_reference_results/*.txt.gz"):
    print("Parsing", path)
    with gzip.open(path) as fhandle:
        for line in fhandle.readlines():
            if OBJ_REGEX.match(line):
                obj_value = Fraction(line.split(' : ')[1])
                the_final_netlib_results[os.path.basename(path).replace('.txt.gz', '').upper()] = {
                    "Objvalue": obj_value}
                break

for key, value in six.iteritems(the_final_netlib_results):
    assert "Objvalue" in value
    assert isinstance(value['Objvalue'], Fraction)

with open('the_final_netlib_results.pcl', 'w') as fhandle:
    pickle.dump(the_final_netlib_results, fhandle, protocol=2)
