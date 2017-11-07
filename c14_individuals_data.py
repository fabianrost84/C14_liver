# coding=UTF-8
import pandas as pd

 
c14_individual_data = pd.read_excel('data/C14data_liver_samples.xlsx')[['sort', 'Code', 'DOB', 'DOA', u'Δ 14C']]
c14_individual_data.columns = ['type', 'pub_id', 'Dbirth', 'Dcoll', 'd14C']
c14_individual_data['mass'] = 1.0
c14_individual_data['d14C'] /= 1000.0

## rounding to avoid interpolation
#c14_individual_data['Dbirth'] = c14_individual_data['Dbirth'].round(1)
#c14_individual_data['Dcoll'] = c14_individual_data['Dcoll'].round(1)

#Index([u'pub_id', u'age', u'Date of Birth', u'Date of Death', u'sex', u'type', u'F14C', u'F14C (SD)', u'Δ14C', u'Δ14C (SD)', u'FACS purity corrected Δ14C', u'NeuN+# (million)', u'NeuN-# (million)', u'Sorting Purity NeuN+ (%)', u'carbon mass according to measured DNA', u'carbon mass measured in graphitization reactor'], dtype='object')
