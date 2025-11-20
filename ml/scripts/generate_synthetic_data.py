import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import hashlib
from faker import Faker
import json
import random

# Valid US city-state pairs for realistic geolocation
VALID_CITY_STATE_PAIRS = [
    ('New York', 'NY'), ('Los Angeles', 'CA'), ('Chicago', 'IL'), ('Houston', 'TX'),
    ('Phoenix', 'AZ'), ('Philadelphia', 'PA'), ('San Antonio', 'TX'), ('San Diego', 'CA'),
    ('Dallas', 'TX'), ('San Jose', 'CA'), ('Austin', 'TX'), ('Jacksonville', 'FL'),
    ('Fort Worth', 'TX'), ('Columbus', 'OH'), ('Charlotte', 'NC'), ('San Francisco', 'CA'),
    ('Indianapolis', 'IN'), ('Austin', 'TX'), ('Seattle', 'WA'), ('Denver', 'CO'),
    ('Boston', 'MA'), ('Miami', 'FL'), ('Portland', 'OR'), ('Atlanta', 'GA'),
    ('Nashville', 'TN'), ('Detroit', 'MI'), ('Minneapolis', 'MN'), ('Memphis', 'TN'),
    ('Baltimore', 'MD'), ('Louisville', 'KY'), ('Milwaukee', 'WI'), ('Albuquerque', 'NM'),
    ('Tucson', 'AZ'), ('Fresno', 'CA'), ('Sacramento', 'CA'), ('Long Beach', 'CA'),
    ('Kansas City', 'MO'), ('Mesa', 'AZ'), ('Virginia Beach', 'VA'), ('Atlanta', 'GA'),
    ('New Orleans', 'LA'), ('Cleveland', 'OH'), ('Plano', 'TX'), ('Arlington', 'TX'),
    ('Las Vegas', 'NV'), ('Corpus Christi', 'TX'), ('Lexington', 'KY'), ('Stockton', 'CA'),
    ('St. Louis', 'MO'), ('Cincinnati', 'OH'), ('Anchorage', 'AK'), ('Henderson', 'NV'),
    ('Chandler', 'AZ'), ('Laredo', 'TX'), ('Lincoln', 'NE'), ('Madison', 'WI'),
]

# Setup logging
FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFMT)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# Configuration
N_USERS = 1000
AVG_CLAIMS_PER_USER = 10  # Will result in ~10,000 claims
FRAUD_RATE = 0.20  # 18% of claims will be fraudulent

# -- Function to generate synthetic users
def generate_synthetic_users(n_users: int = N_USERS) -> pd.DataFrame:
    """
    Generate synthetic user data with varying risk profiles

    Returns:
        pd.DataFrame: Users dataframe
    """
    logger.info(f"Generating {n_users} synthetic users...")

    users = []

    for i in range(n_users):
        # For varying risk profiles
        # Risk profile distribution: 70% low, 20% medium, 10% high
        risk_profile = np.random.choice(
            ['low', 'medium', 'high'],
            p=[0.70, 0.20, 0.10]
        )

        # Policy start date (1-5 years ago)
        policy_start_date = fake.date_between(start_date='-5y', end_date='-1m')
        account_age_days = (datetime.now().date() - policy_start_date).days

        # Policy type
        policy_type = np.random.choice(
            ['auto', 'home', 'health', 'life'],
            p=[0.40, 0.25, 0.25, 0.10]
        )

        # Coverage and premium based on policy type
        if policy_type == 'auto':
            coverage = np.random.uniform(50000, 300000)
            premium = coverage * np.random.uniform(0.01, 0.025)
        elif policy_type == 'home':
            coverage = np.random.uniform(200000, 800000)
            premium = coverage * np.random.uniform(0.005, 0.015)
        elif policy_type == 'health':
            coverage = np.random.uniform(100000, 1000000)
            premium = coverage * np.random.uniform(0.02, 0.04)
        else:  # life
            coverage = np.random.uniform(100000, 2000000)
            premium = coverage * np.random.uniform(0.005, 0.02)

        # Fraud propensity based on risk profile
        if risk_profile == 'low':
            fraud_propensity = np.random.uniform(0.02, 0.08)
            claim_frequency = 'low'
        elif risk_profile == 'medium':
            fraud_propensity = np.random.uniform(0.12, 0.20)
            claim_frequency = 'medium'
        else:  # high
            fraud_propensity = np.random.uniform(0.25, 0.40)
            claim_frequency = 'high'

        # Select a valid city-state pair for user location
        user_city, user_state = VALID_CITY_STATE_PAIRS[np.random.randint(len(VALID_CITY_STATE_PAIRS))]
        
        user = {
            # user biodata
            'user_id': str(uuid.uuid4()),
            'email': f'user{i:04d}@example.com',
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'phone_number': fake.phone_number(),
            'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=75),
            'address': fake.street_address(),
            'city': user_city,
            'state': user_state,
            'country': 'USA',
            'postal_code': fake.zipcode(),

            # Policy details
            'policy_number': f'POL-{i:06d}',
            'policy_type': policy_type,
            'policy_start_date': policy_start_date,
            'policy_end_date': policy_start_date + timedelta(days=365 * np.random.randint(1, 3)),
            'premium_amount': round(premium, 2),
            'coverage_amount': round(coverage, 2),

            # Account metadata
            'account_created_at': policy_start_date,
            'account_age_days': account_age_days,
            'account_status': 'active',
            'is_verified': True,

            # Risk profile
            'risk_category': risk_profile,
            'claim_frequency': claim_frequency,
            'fraud_propensity': fraud_propensity,

            # These will be updated after claim generation
            'total_claims_count': 0,
            'total_claims_amount': 0.0,
            'fraud_flags_count': 0
        }

        users.append(user)

    users_df = pd.DataFrame(users)
    logger.info(f"✓ Generated {len(users_df)} users")
    print(f"  - Low risk: {len(users_df[users_df['risk_category'] == 'low'])}")
    print(f"  - Medium risk: {len(users_df[users_df['risk_category'] == 'medium'])}")
    print(f"  - High risk: {len(users_df[users_df['risk_category'] == 'high'])}")

    return users_df

generate_synthetic_users()

# -- Function to generate claim descriptions
def generate_claim_description(claim_type: str, is_fraud: bool) -> str:
    """Generate realistic claim descriptions"""

    descriptions = {
        'accident': [
            "Rear-ended at traffic light, significant damage to rear bumper and trunk",
            "Side-swiped while changing lanes on highway, damage to passenger side doors",
            "Hit by shopping cart in parking lot, minor dent and paint damage",
            "Collision with deer on rural road, front end damage",
            "Multi-vehicle pileup on icy road, extensive front and side damage"
        ],
        'theft': [
            "Vehicle stolen from driveway overnight, recovered by police damaged",
            "Laptop and electronics stolen during home break-in",
            "Catalytic converter stolen from vehicle, requires replacement",
            "Jewelry stolen during vacation, police report filed",
            "Tools and equipment stolen from garage"
        ],
        'medical': [
            "Emergency room visit for chest pain and cardiac evaluation",
            "Surgical procedure for torn ACL from sports injury",
            "Treatment for broken arm from fall, includes X-rays and casting",
            "Hospitalization for severe pneumonia, 3-day stay",
            "Physical therapy sessions for lower back injury"
        ],
        'property_damage': [
            "Water damage from burst pipe in basement, flooring and drywall affected",
            "Wind damage to roof shingles during severe storm",
            "Fire damage to kitchen from electrical fault, requires renovation",
            "Tree fell on garage during storm, structural damage",
            "Hail damage to roof and siding"
        ],
        'other': [
            "Personal liability claim from slip and fall on property",
            "Damage to fence from neighbor's tree",
            "Lost personal items during travel",
            "Damage to property from contractor negligence",
            "Legal defense costs for liability claim"
        ]
    }

    base_description = np.random.choice(descriptions.get(claim_type, descriptions['other']))

    # Fraudulent claims sometimes have unclear or overly detailed descriptions
    if is_fraud and np.random.random() < 0.3:
        base_description += " [Requires additional verification and documentation]"

    return base_description

# -- Function to generate synthetic claims
def generate_synthetic_claims(user_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic claims with realistic fraud patterns

    Args:
        users_df: DataFrame of users

    Returns:
        pd.DataFrame: Claims dataframe
    """
    logger.info(f"\nGenerating claims (targeting ~10,000 claims)...")

    claims = []
    claim_counter = 0


    # Generate claims for users
    for _, user in user_df.iterrows():
        # Determine number of claims based on user claim frequency
        if user['claim_frequency'] == 'low':
            n_claims = np.random.poisson(7)
        elif user['claim_frequency'] == 'medium':
            n_claims = np.random.poisson(12)
        else:  # high
            n_claims = np.random.poisson(20)

        # Each user can have a maximum of 25 claims
        # Ensure at least 1 claim per user (1-25 claims per user)
        n_claims = max(1, min(n_claims, 25))

        user_claims = []

        for claim_idx in range(n_claims):
            # Determine if this claim is fraudulent
            is_fraud = np.random.random() < user['fraud_propensity']

            # Claim type based on policy type with some variation
            if user['policy_type'] == 'auto':
                claim_type = np.random.choice(
                    ['accident', 'theft', 'other'],
                    p=[0.70, 0.20, 0.10]
                )
            elif user['policy_type'] == 'home':
                claim_type = np.random.choice(
                    ['property_damage', 'theft', 'other'],
                    p=[0.60, 0.25, 0.15]
                )
            elif user['policy_type'] == 'health':
                claim_type = np.random.choice(
                    ['medical', 'other'],
                    p=[0.90, 0.10]
                )
            else:  # life
                claim_type = 'other'

            # Generate base claim amount based on claim type
            if claim_type == 'medical':
                base_amount = np.random.lognormal(8, 1.2)  # $1k - $400k
            elif claim_type == 'accident':
                base_amount = np.random.lognormal(7.5, 1.0)  # $800 - $150k
            elif claim_type == 'property_damage':
                base_amount = np.random.lognormal(7, 1.3)  # $500 - $200k
            elif claim_type == 'theft':
                base_amount = np.random.lognormal(6.5, 1.1)  # $300 - $80k
            else:  # other
                base_amount = np.random.lognormal(6, 1.0)  # $200 - $50k

            # Ensure the claim does not exceed coverage (with some margin)
            base_amount = min(base_amount, user['coverage_amount'] * 0.95)

            fraud_type = None

            # Apply fraud patterns
            if is_fraud:
                fraud_pattern = np.random.choice([
                    'exaggerated',
                    'duplicate',
                    'staged',
                    'identity_theft',
                    'out_of_pattern'
                ], p=[0.35, 0.15, 0.25, 0.10, 0.15])

                if fraud_pattern == 'exaggerated':
                    # Exaggerate claim amount by 1.5x to 3x
                    claim_amount = base_amount * np.random.uniform(1.5, 3.0)
                    fraud_type = 'exaggerated'

                elif fraud_pattern == 'duplicate' and len(user_claims) > 0:
                    # Duplicate a previous claim with slight variation
                    prev_claim = np.random.choice(user_claims)
                    claim_amount = prev_claim['claim_amount'] * np.random.uniform(0.9, 1.1)
                    claim_type = prev_claim['claim_type']
                    fraud_type = 'duplicate'

                elif fraud_pattern == 'staged':
                    # Staged incidents often have round numbers
                    claim_amount = round(base_amount / 1000) * 1000
                    fraud_type = 'staged'

                elif fraud_pattern == 'identity_theft':
                    claim_amount = base_amount
                    fraud_type = 'identity_theft'

                else:  # out_of_pattern
                    claim_amount = base_amount * np.random.uniform(1.2, 2.0)
                    fraud_type = 'out_of_pattern'
            else:
                claim_amount = base_amount

            # Incident date (has to be within policy period)
            earliest_date = user['policy_start_date']
            latest_date = min(
                datetime.now().date(),
                user['policy_end_date']
            )

            days_diff = (latest_date - earliest_date).days
            if days_diff > 0:
                incident_date = earliest_date + timedelta(days=np.random.randint(0, days_diff))
            else:
                incident_date = earliest_date

            # Submission timestamp (0-60 days after incident)
            days_delay = np.random.randint(0, 61)

            submitted_at = datetime.combine(
                incident_date + timedelta(days=days_delay),
                datetime.min.time()
            ) + timedelta(
                hours=np.random.randint(6, 23),
                minutes=np.random.randint(0, 60)
            )

            # Supporting documents count
            if is_fraud and fraud_type == 'staged':
                # Staged fraud often has fewer documents
                docs_count = np.random.randint(0, 3)
            else:
                docs_count = np.random.randint(1, 6)

            # Fraud confidence score (ground truth for evaluation)
            if is_fraud:
                fraud_confidence = np.random.uniform(0.70, 0.98)
            else:
                fraud_confidence = np.random.uniform(0.0, 0.25)

            # Claim status
            if is_fraud and fraud_confidence > 0.85:
                claim_status = np.random.choice(
                    ['pending', 'under_review', 'rejected'],
                    p=[0.3, 0.5, 0.2]
                )
            else:
                claim_status = np.random.choice(
                    ['pending', 'approved', 'under_review'],
                    p=[0.4, 0.5, 0.1]
                )

            claim = {
                'claim_id': str(uuid.uuid4()),
                'user_id': user['user_id'],
                'claim_number': f'CLM-{claim_counter:08d}',
                'claim_type': claim_type,
                'claim_amount': round(claim_amount, 2),
                'incident_date': incident_date,
                'claim_description': generate_claim_description(claim_type, is_fraud),
                'supporting_documents_count': docs_count,
                'claim_status': claim_status,
                'submitted_at': submitted_at,
                'processed_at': None if claim_status == 'pending' else submitted_at + timedelta(
                    days=np.random.randint(1, 15)),

                # Ground truth labels
                'is_fraudulent': is_fraud,
                'fraud_type': fraud_type,
                'fraud_confidence_score': fraud_confidence,

                # Approval details
                'approval_status': None if claim_status == 'pending' else claim_status,
                'rejection_reason': 'Suspected fraud' if claim_status == 'rejected' else None,
                'approved_amount': round(claim_amount * 0.95, 2) if claim_status == 'approved' else None
            }

            claims.append(claim)
            user_claims.append(claim)
            claim_counter += 1

    claims_df = pd.DataFrame(claims)

    # Sort by submission time
    claims_df = claims_df.sort_values('submitted_at').reset_index(drop=True)

    logger.info(f"✓ Generated {len(claims_df)} claims")
    print(f"  - Fraudulent: {claims_df['is_fraudulent'].sum()} ({claims_df['is_fraudulent'].mean() * 100:.1f}%)")
    print(f"  - Legitimate: {(~claims_df['is_fraudulent']).sum()} ({(~claims_df['is_fraudulent']).mean() * 100:.1f}%)")

    fraud_types = claims_df[claims_df['is_fraudulent']]['fraud_type'].value_counts()

    print(f"\n  Fraud type distribution:")
    for fraud_type, count in fraud_types.items():
        print(f"    - {fraud_type}: {count}")

    return claims_df

# -- Function to generate transaction context for claims
def generate_transaction_contexts(
    claims_df: pd.DataFrame, users_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate transaction context data for each claim

    Args:
        claims_df: DataFrame of claims
        users_df: DataFrame of users

    Returns:
        pd.DataFrame: Transaction contexts dataframe
    """
    logger.info(f"\nGenerating transaction contexts...")

    contexts = []

    # Create user device mapping (most users have 1-2 trusted devices)
    user_devices = {}
    for user_id in users_df['user_id']:
        n_devices = np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10])
        devices = []

        for _ in range(n_devices):
            device_type = np.random.choice(
                ['mobile', 'desktop', 'tablet'],
                p=[0.60, 0.35, 0.05]
            )
            devices.append({
                'type': device_type,
                'fingerprint': hashlib.md5(f"{user_id}-{device_type}-{uuid.uuid4()}".encode()).hexdigest()[:16],
                'first_seen': datetime.now() - timedelta(days=np.random.randint(30, 1000))
            })

        user_devices[user_id] = devices

    # Create user location mapping
    user_locations = users_df.set_index('user_id')[['city', 'state']].to_dict('index')

    for _, claim in claims_df.iterrows():
        user = users_df[users_df['user_id'] == claim['user_id']].iloc[0]

        user_claim_history = claims_df[
            (claims_df['user_id'] == claim['user_id']) &
            (claims_df['submitted_at'] < claim['submitted_at'])
        ].sort_values('submitted_at')

        # Device selection
        is_fraud_identity_theft = claim['is_fraudulent'] and claim['fraud_type'] == 'identity_theft'

        if is_fraud_identity_theft:
            # Identity theft uses unknown device
            device_type = np.random.choice(['mobile', 'desktop', 'tablet'])
            device_fingerprint = hashlib.md5(f"unknown-{uuid.uuid4()}".encode()).hexdigest()[:16]
            is_trusted_device = False
            device_trust_score = np.random.uniform(0.0, 0.2)
        else:
            # Normal: use one of user's known devices
            device = np.random.choice(user_devices[claim['user_id']])
            device_type = device['type']
            device_fingerprint = device['fingerprint']
            is_trusted_device = True
            device_trust_score = np.random.uniform(0.8, 1.0)

        # Geolocation
        if is_fraud_identity_theft:
            # Identity theft: unusual location with valid city-state pair
            geolocation_distance = np.random.uniform(500, 3000)
            is_geolocation_anomaly = True
            location_city, location_state = VALID_CITY_STATE_PAIRS[np.random.randint(len(VALID_CITY_STATE_PAIRS))]
        else:
            # Normal: nearby location
            geolocation_distance = np.random.uniform(0, 50)
            is_geolocation_anomaly = False
            location_city = user_locations[claim['user_id']]['city']
            location_state = user_locations[claim['user_id']]['state']

        # Session behavior
        if claim['is_fraudulent']:
            # Fraudsters often rush or behave abnormally
            if np.random.random() < 0.6:
                # Rushed
                form_fill_time = np.random.uniform(30, 180)
                session_duration = np.random.uniform(120, 600)
            else:
                # Overly careful
                form_fill_time = np.random.uniform(900, 1800)
                session_duration = np.random.uniform(1200, 3600)
        else:
            # Normal behavior
            form_fill_time = np.random.uniform(180, 900)
            session_duration = np.random.uniform(300, 1800)

        # Time patterns
        hour = claim['submitted_at'].hour
        day_of_week = claim['submitted_at'].weekday()
        is_unusual_time = (hour < 6 or hour > 22)

        # Days since last claim
        if len(user_claim_history) > 0:
            last_claim = user_claim_history.iloc[-1]
            days_since_last = (claim['submitted_at'] - last_claim['submitted_at']).days
            is_rapid_succession = days_since_last < 7
        else:
            days_since_last = None
            is_rapid_succession = False

        # IP address (simulate realistic IPs)
        if is_fraud_identity_theft:
            ip_address = fake.ipv4_public()
        else:
            # Same subnet for legitimate users
            base_ip = f"192.168.{np.random.randint(0, 255)}"
            ip_address = f"{base_ip}.{np.random.randint(1, 255)}"

        context = {
            'context_id': str(uuid.uuid4()),
            'claim_id': claim['claim_id'],
            'user_id': claim['user_id'],
            'device_id': device_fingerprint,

            # Device info
            'device_type': device_type,
            'device_fingerprint': device_fingerprint,
            'is_trusted_device': is_trusted_device,
            'device_trust_score': round(device_trust_score, 4),
            'os': np.random.choice(['Windows', 'macOS', 'Linux', 'iOS', 'Android']),
            'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),

            # Location
            'ip_address': ip_address,
            'geolocation_city': location_city,
            'geolocation_state': location_state,
            'geolocation_country': 'USA',
            'geolocation_distance_km': round(geolocation_distance, 2),
            'is_geolocation_anomaly': is_geolocation_anomaly,

            # Session behavior
            'session_duration_seconds': int(session_duration),
            'form_fill_time_seconds': int(form_fill_time),
            'pages_visited': np.random.randint(3, 15),

            # Time patterns
            'transaction_hour': hour,
            'transaction_day_of_week': day_of_week,
            'is_unusual_time': is_unusual_time,

            # Claim patterns
            'days_since_last_claim': days_since_last,
            'is_rapid_succession': is_rapid_succession,
            'claims_last_30_days': len(
                user_claim_history[
                    user_claim_history['submitted_at'] >= claim['submitted_at'] - timedelta(
                        days=30)
                    ]),
            'claims_last_90_days': len(
                user_claim_history[
                    user_claim_history['submitted_at'] >= claim['submitted_at'] - timedelta(
                        days=90)
                    ]),

            'captured_at': claim['submitted_at']
        }

        contexts.append(context)

    contexts_df = pd.DataFrame(contexts)

    logger.info(f"✓ Generated {len(contexts_df)} transaction contexts")
    print(f"  - Trusted devices: {contexts_df['is_trusted_device'].sum()}")
    print(f"  - Geolocation anomalies: {contexts_df['is_geolocation_anomaly'].sum()}")
    print(f"  - Rapid succession: {contexts_df['is_rapid_succession'].sum()}")

    return contexts_df

# -- Function to create full synthetic dataset
def create_synthetic_dataset(
    users_df: pd.DataFrame, claims_df: pd.DataFrame, contexts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a merged dataset for ML model training

    Args:
        users_df: Users dataframe
        claims_df: Claims dataframe
        contexts_df: Transaction contexts dataframe

    Returns:
        pd.DataFrame: Merged dataset with all features
    """
    logger.info(f"\nCreating merged Synthetic dataset...")

    # Merge claims with contexts
    synthetic_df = claims_df.merge(contexts_df, on='claim_id', how='left')

    # user_id feature changes to user_id_x after merge.
    # Return it to user_id to be able to merge user features later on
    synthetic_df.rename(columns={"user_id_x": "user_id"}, inplace=True)

    # Add user features
    user_features = users_df[[
        'user_id', 'policy_type', 'policy_start_date', 'premium_amount',
        'coverage_amount', 'account_age_days', 'risk_category'
    ]]

    synthetic_df = synthetic_df.merge(user_features, on='user_id', how='left')

    # -- Feature engineering
    # Creating some features for better relationship representation
    synthetic_df['claim_amount_ratio'] = synthetic_df['claim_amount'] / synthetic_df['coverage_amount']

    # Convert policy_start_date to datetime
    synthetic_df['policy_start_date'] = pd.to_datetime(synthetic_df['policy_start_date'])

    synthetic_df['days_since_policy_start'] = (
        synthetic_df['submitted_at'] - synthetic_df['policy_start_date']
    ).dt.days
    synthetic_df['is_weekend'] = synthetic_df['transaction_day_of_week'].isin([5, 6])

    # Calculate user claim statistics up to each claim
    synthetic_df['user_avg_claim_amount'] = synthetic_df.groupby('user_id')['claim_amount'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.mean())
    )
    synthetic_df['user_total_claims'] = synthetic_df.groupby('user_id').cumcount()

    logger.info(f"✓ Created ML dataset with {len(synthetic_df)} rows and {len(synthetic_df.columns)} columns")

    return synthetic_df

# -- Function to update user statistics
def update_user_statistics(
    users_df: pd.DataFrame, claims_df: pd.DataFrame
) -> pd.DataFrame:
    """Update user statistics based on claims."""
    logger.info(f"\nUpdating user statistics...")

    user_stats = claims_df.groupby('user_id').agg({
        'claim_amount': ['count', 'sum'],
        'is_fraudulent': 'sum'
    }).reset_index()

    user_stats.columns = ['user_id', 'total_claims_count', 'total_claims_amount', 'fraud_flags_count']

    users_df = users_df.merge(user_stats, on='user_id', how='left', suffixes=('', '_new'))
    users_df['total_claims_count'] = users_df['total_claims_count_new'].fillna(0).astype(int)
    users_df['total_claims_amount'] = users_df['total_claims_amount_new'].fillna(0)
    users_df['fraud_flags_count'] = users_df['fraud_flags_count_new'].fillna(0).astype(int)

    users_df = users_df.drop(columns=['total_claims_count_new', 'total_claims_amount_new', 'fraud_flags_count_new'])

    logger.info(f"✓ Updated user statistics")

    return users_df

# -- Function to save datasets
def save_datasets(
    users_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    contexts_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    output_dir: str = '../data/synthetic'
) -> None:
    """Save all datasets to CSV files."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\nSaving datasets to {output_dir}...")

    users_df.to_csv(f'{output_dir}/users.csv', index=False)
    print(f"✓ Saved users.csv ({len(users_df)} rows)")

    claims_df.to_csv(f'{output_dir}/claims.csv', index=False)
    print(f"✓ Saved claims.csv ({len(claims_df)} rows)")

    contexts_df.to_csv(f'{output_dir}/transaction_contexts.csv', index=False)
    print(f"✓ Saved transaction_contexts.csv ({len(contexts_df)} rows)")

    synthetic_df.to_csv(f'{output_dir}/insurance_claims_v1_full.csv', index=False)
    print(f"✓ Saved ml_dataset.csv ({len(synthetic_df)} rows)")

    # Save data summary
    summary = {
        'generation_date': datetime.now().isoformat(),
        'n_users': len(users_df),
        'n_claims': len(claims_df),
        'fraud_rate': float(claims_df['is_fraudulent'].mean()),
        'total_claim_amount': float(claims_df['claim_amount'].sum()),
        'avg_claim_amount': float(claims_df['claim_amount'].mean()),
        'risk_distribution': users_df['risk_category'].value_counts().to_dict(),
        'fraud_type_distribution': claims_df[claims_df['is_fraudulent']]['fraud_type'].value_counts().to_dict()
    }

    with open(f'{output_dir}/data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved data_summary.json")
    print(f"\n{'=' * 60}")
    print(f"DATA GENERATION COMPLETE")
    print(f"{'=' * 60}")

# -- Entry point function
def main():
    """Main execution function."""
    print(f"\n{'=' * 60}")
    print(f"INSURANCE FRAUD DETECTION - SYNTHETIC DATA GENERATION")
    print(f"{'=' * 60}\n")

    # Generate users
    users_df = generate_synthetic_users(N_USERS)

    # Generate claims
    claims_df = generate_synthetic_claims(users_df)

    # Generate transaction contexts
    contexts_df = generate_transaction_contexts(claims_df, users_df)

    # Create full synthetic dataset
    synthetic_df = create_synthetic_dataset(users_df, claims_df, contexts_df)

    # Update user statistics
    users_df = update_user_statistics(users_df, claims_df)

    # Save all datasets
    save_datasets(users_df, claims_df, contexts_df, synthetic_df)

    # Print final summary
    print(f"\nFINAL SUMMARY:")
    print(f"  - Users: {len(users_df)}")
    print(f"  - Claims: {len(claims_df)}")
    print(f"  - Fraud rate: {claims_df['is_fraudulent'].mean() * 100:.1f}%")
    print(f"  - Avg claims per user: {len(claims_df) / len(users_df):.1f}")
    print(f"  - Total claim amount: ${claims_df['claim_amount'].sum():,.2f}")
    print(f"\nDatasets saved to: ml/data/synthetic/")

# -- Run script
if __name__ == "__main__":
    main()
