# LumeNN

**LumeNN** is a neural network that solves problems of binary and multi-class classification of stellar luminosity.


## Binary classification

### Dataset

Dataset was extracted from **ATLAS-REFCAT2** catalog by using [TAPVizieR](https://tapvizier.cds.unistra.fr/adql)
webservice.

[Full dataset](https://archive.stsci.edu/hlsp/atlas-refcat2) has almost 1 billion entries 
(*992637834* to be exact), which is way too much to process.
[TAPVizieR](https://tapvizier.cds.unistra.fr/adql) allowed to query only those entries,
who have **dupvar** (star luminosity) set to *2* (constant) or *1* (variable).

Research was done on *200000* dataset with *100000* samples of stars with constant luminosity
and *100000* samples of stars with variable luminosity.
You can see those samples in [stellardata.sqlite](./data/stellardata.sqlite) database
in the **binary_classification** table.

'Clean up' of data was also performed - columns were renamed, types fixed, `NULL`s added.
[Here](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/ApJ/867/105/refcat2) you can see
extensive information about initial state of the dataset.
