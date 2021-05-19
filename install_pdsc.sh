cd pdsc
pip install .
mkdir pdsc_tables
cd pdsc_tables
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB -o RDRCUMINDEX.TAB
curl https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL -o RDRCUMINDEX.LBL
cd ..
pdsc_ingest ./pdsc_tables/RDRCUMINDEX.LBL ./pdsc_tables