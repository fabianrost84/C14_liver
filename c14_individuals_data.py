# coding=UTF-8
import pandas as pd

 
c14_individual_data = pd.read_excel('./mmc1.xlsx', na_values = ['ND', 'NA'])


c14_individual_data = c14_individual_data[['type', 'pub_id', 'Date of Birth', 'Date of Death', u'FACS purity corrected Δ14C', u'carbon mass according to measured DNA']]

c14_individual_data.columns = ['type', 'pub_id', 'Dbirth', 'Dcoll', 'd14C', 'mass']
c14_individual_data['d14C'] /= 1000.0

## rounding to avoid interpolation
#c14_individual_data['Dbirth'] = c14_individual_data['Dbirth'].round(1)
#c14_individual_data['Dcoll'] = c14_individual_data['Dcoll'].round(1)

c14_neu = c14_individual_data[ c14_individual_data['type'] == u'neuronal nuclei (NeuN+)']
c14_nonneu = c14_individual_data[ c14_individual_data['type'] == u'non-neuronal nuclei (NeuN-)']

#Index([u'pub_id', u'age', u'Date of Birth', u'Date of Death', u'sex', u'type', u'F14C', u'F14C (SD)', u'Δ14C', u'Δ14C (SD)', u'FACS purity corrected Δ14C', u'NeuN+# (million)', u'NeuN-# (million)', u'Sorting Purity NeuN+ (%)', u'carbon mass according to measured DNA', u'carbon mass measured in graphitization reactor'], dtype='object')
