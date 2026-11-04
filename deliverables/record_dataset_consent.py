"""Record the project owner's explicit report; not individual signed evidence."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record_all():
    path = ROOT/'deliverables/dataset-candidates.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    stamp = datetime.now(timezone.utc).isoformat()
    manifest = {'reported_at': stamp, 'reported_date_moscow': '2026-11-04',
                'evidence': {'type': 'project_owner_report_in_chat',
                             'text': 'получил согласия от всех, продолжай',
                             'individual_confirmation_files': False},
                'scopes_not_assumed': ['local_client_files', 'replay_files', 'public_redistribution'],
                'players': [{'user_id': r['user_id'], 'username': r['username'], 'status': 'granted',
                             'scopes': ['public_scores_training_evaluation']} for r in rows]}
    folder = ROOT/'.data/population'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'consents.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in rows: row['consent'] = 'owner_confirmed'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
        writer.writeheader(); writer.writerows(rows)
    summary_path = ROOT/'deliverables/dataset-candidates-summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['consent_report'] = {'players': len(rows), 'date': '2026-11-04', 'evidence': 'project_owner_report_in_chat'}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Recorded owner-reported consent for {len(rows)} players, public scores for training and evaluation only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-all', action='store_true', required=True)
    parser.parse_args()
    record_all()
