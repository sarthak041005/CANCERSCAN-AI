from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets'
REPORTS = ROOT / 'reports'
TRAIN_CSV = DATA / 'train.csv'
VAL_CSV = DATA / 'val.csv'
KEY_FIELDS = ['patient_id', 'breast', 'abnormality_type', 'pathology', 'assessment', 'subtlety']


def norm(x):
    if pd.isna(x):
        return ''
    return str(x).strip().upper()


def load(path, split):
    if not path.exists():
        raise FileNotFoundError(f'Missing CSV: {path}')
    df = pd.read_csv(path)
    required = set(KEY_FIELDS + ['image_path', 'view'])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{path.name} missing columns: {missing}. Available: {list(df.columns)}')
    if 'label' not in df.columns and 'pathology' not in df.columns:
        raise ValueError(f'{path.name} needs label or pathology column')
    for c in KEY_FIELDS + ['view']:
        df[c + '_norm'] = df[c].map(norm)
    if 'label' in df.columns:
        def label(row):
            v = row.get('label')
            if pd.notna(v):
                try:
                    i = int(float(v))
                    if i in (0, 1): return i
                except (ValueError, TypeError):
                    pass
            p = norm(row.get('pathology', ''))
            if p.startswith('BENIGN'): return 0
            if p.startswith('MALIGNANT'): return 1
            return -1
        df['label_norm'] = df.apply(label, axis=1)
    else:
        df['label_norm'] = df['pathology'].map(lambda x: 0 if norm(x).startswith('BENIGN') else 1 if norm(x).startswith('MALIGNANT') else -1)
    df['split'] = split
    df['view_norm'] = df['view'].map(norm)
    return df


def main():
    REPORTS.mkdir(exist_ok=True)
    tr = load(TRAIN_CSV, 'train')
    va = load(VAL_CSV, 'validation')
    overlap = sorted(set(tr['patient_id_norm']) & set(va['patient_id_norm']))
    summaries = {}
    all_unmatched = []
    all_ambiguous = []
    pair_key_cols = [c + '_norm' for c in KEY_FIELDS] + ['label_norm']

    for name, df in [('train', tr), ('validation', va)]:
        counts = {'split': name, 'input_rows': int(len(df)), 'cc_rows': int((df.view_norm == 'CC').sum()), 'mlo_rows': int((df.view_norm == 'MLO').sum()), 'other_view_rows': int((~df.view_norm.isin(['CC','MLO'])).sum()), 'paired_groups': 0, 'unmatched_cc_rows': 0, 'unmatched_mlo_rows': 0, 'ambiguous_groups': 0, 'invalid_labels': int((df.label_norm == -1).sum())}
        for _, g in df.groupby(pair_key_cols, dropna=False, sort=False):
            cc = g[g.view_norm == 'CC']
            mlo = g[g.view_norm == 'MLO']
            if len(cc) == 1 and len(mlo) == 1:
                counts['paired_groups'] += 1
            elif len(cc) and not len(mlo):
                counts['unmatched_cc_rows'] += len(cc)
                for _, r in cc.iterrows():
                    all_unmatched.append({'split': name, 'unmatched_view': 'CC', **{c: r.get(c, '') for c in KEY_FIELDS}, 'label': r.get('label',''), 'image_path': r.get('image_path',''), 'patient_breast_group_cc_count': int(len(cc)), 'patient_breast_group_mlo_count': 0})
            elif len(mlo) and not len(cc):
                counts['unmatched_mlo_rows'] += len(mlo)
                for _, r in mlo.iterrows():
                    all_unmatched.append({'split': name, 'unmatched_view': 'MLO', **{c: r.get(c, '') for c in KEY_FIELDS}, 'label': r.get('label',''), 'image_path': r.get('image_path',''), 'patient_breast_group_cc_count': 0, 'patient_breast_group_mlo_count': int(len(mlo))})
            elif len(cc) or len(mlo):
                counts['ambiguous_groups'] += 1
                for _, r in g.iterrows():
                    all_ambiguous.append({'split': name, **{c: r.get(c, '') for c in KEY_FIELDS}, 'view': r.get('view',''), 'label': r.get('label',''), 'image_path': r.get('image_path',''), 'group_cc_count': len(cc), 'group_mlo_count': len(mlo)})

        # For each unmatched record, show other records from same patient and breast that might explain mismatch.
        summaries[name] = counts

    unmatched_df = pd.DataFrame(all_unmatched)
    if not unmatched_df.empty:
        mismatch_rows = []
        for _, u in unmatched_df.iterrows():
            base = (tr if u['split'] == 'train' else va)
            same = base[(base['patient_id_norm'] == norm(u['patient_id'])) & (base['breast_norm'] == norm(u['breast'])) & (base['view_norm'] != u['unmatched_view'])]
            for _, other in same.iterrows():
                diffs = []
                for f in ['abnormality_type','pathology','assessment','subtlety']:
                    if norm(u.get(f, '')) != norm(other.get(f, '')):
                        diffs.append(f)
                if 'label_norm' in other and int(other['label_norm']) != int(u.get('label', other['label_norm']) if str(u.get('label','')).strip() not in ('','nan') else other['label_norm']):
                    diffs.append('label')
                mismatch_rows.append({'split': u['split'], 'patient_id': u['patient_id'], 'breast': u['breast'], 'unmatched_view': u['unmatched_view'], 'unmatched_image_path': u['image_path'], 'candidate_other_view': other.get('view',''), 'candidate_image_path': other.get('image_path',''), 'candidate_abnormality_type': other.get('abnormality_type',''), 'candidate_pathology': other.get('pathology',''), 'candidate_assessment': other.get('assessment',''), 'candidate_subtlety': other.get('subtlety',''), 'candidate_label': other.get('label',''), 'fields_that_differ': ', '.join(diffs) if diffs else 'none; inspect duplicate/grouping context'})
        mismatch_df = pd.DataFrame(mismatch_rows)
    else:
        mismatch_df = pd.DataFrame(columns=['split','patient_id','breast','unmatched_view','unmatched_image_path','candidate_other_view','candidate_image_path','fields_that_differ'])

    audit = {'train': summaries['train'], 'validation': summaries['validation'], 'patient_overlap_count': len(overlap), 'patient_overlap_examples': overlap[:20], 'important_note': 'This is a diagnostic audit only. Candidate CC/MLO records sharing patient and breast are NOT automatically declared valid pairs; abnormality/lesion metadata must establish a defensible match.'}
    (REPORTS / 'breast_cc_mlo_pairing_diagnostic.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    unmatched_df.to_csv(REPORTS / 'breast_cc_mlo_unmatched_rows.csv', index=False)
    mismatch_df.to_csv(REPORTS / 'breast_cc_mlo_candidate_mismatches.csv', index=False)
    pd.DataFrame(all_ambiguous).to_csv(REPORTS / 'breast_cc_mlo_ambiguous_groups.csv', index=False)
    print(json.dumps(audit, indent=2))
    print('\nSaved diagnostic files:')
    for n in ['breast_cc_mlo_pairing_diagnostic.json','breast_cc_mlo_unmatched_rows.csv','breast_cc_mlo_candidate_mismatches.csv','breast_cc_mlo_ambiguous_groups.csv']:
        print(REPORTS / n)
    print('\nDo not change pair matching rules based only on patient/breast similarity. Review candidate mismatches against the original CBIS-DDSM metadata first.')

if __name__ == '__main__':
    main()
