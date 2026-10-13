"""Retrieve the primary sources required to rebuild the new array inputs.

Only platform annotations up to !platform_table_end are extracted from GEO
platform families. The large associated sample cache is not downloaded.
"""
from pathlib import Path
import argparse, urllib.request, gzip, hashlib, json, csv

def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ExprBench-PDAC/1.1.0'}),timeout=120)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    out=a.root/'revision_20261012/sources';out.mkdir(parents=True,exist_ok=True)
    tasks={
      'GSE57495_series_matrix.txt.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE57nnn/GSE57495/matrix/GSE57495_series_matrix.txt.gz',
      'E-MTAB-6134.sdrf.txt':'https://www.ebi.ac.uk/biostudies/files/E-MTAB-6134/E-MTAB-6134.sdrf.txt',
      'E-MTAB-6134.idf.txt':'https://www.ebi.ac.uk/biostudies/files/E-MTAB-6134/E-MTAB-6134.idf.txt'}
    for name,url in tasks.items():
        dest=out/name
        if not dest.exists():
            with request(url) as response: dest.write_bytes(response.read())
        print(name,dest.stat().st_size,flush=True)
    for gpl,prefix in [('GPL15048','GPL15nnn'),('GPL13667','GPL13nnn')]:
        dest=out/(gpl+'_platform_only.soft.gz')
        if dest.exists(): continue
        url=f'https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/{gpl}/soft/{gpl}_family.soft.gz'
        lines=[];size=0
        with request(url) as response,gzip.GzipFile(fileobj=response) as source:
            for line in source:
                lines.append(line);size+=len(line)
                if line.startswith(b'!platform_table_end'): break
                if size>350*1024*1024: raise ValueError('Platform annotation extraction bound exceeded')
        assert lines[-1].startswith(b'!platform_table_end')
        dest.write_bytes(gzip.compress(b''.join(lines)))
    dest=out/'ProcessedExpression.tsv.gz'
    if not dest.exists():
        sha=hashlib.sha256();size=0
        with request('https://www.ebi.ac.uk/biostudies/files/E-MTAB-6134/ProcessedExpression.tsv') as response,gzip.open(dest,'wb',compresslevel=3) as target:
            for block in iter(lambda:response.read(64*1024),b''):
                sha.update(block);size+=len(block);target.write(block)
        assert size==258465074 and sha.hexdigest()=='5f241ced40d474649fd660fcdc080714f88fbd2a62ffd33e97a5a93536cd048e'
        (out/'ProcessedExpression_provenance.json').write_text(json.dumps({'uncompressed_bytes':size,'source_sha256':sha.hexdigest()},indent=2))
    header={}
    with gzip.open(out/'GSE57495_series_matrix.txt.gz','rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):break
            if line.startswith(('!Sample_geo_accession','!Sample_title','!Sample_characteristics')):
                row=next(csv.reader([line.rstrip('\n')],delimiter='\t'));header.setdefault(row[0],[]).append(row[1:])
    (out/'GSE57495_metadata.json').write_text(json.dumps(header,indent=2),encoding='utf-8')

if __name__=='__main__':main()
