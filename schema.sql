-- ChronoSense AI — Supabase schema
-- Run this in the Supabase SQL editor (Dashboard → SQL Editor → New query).

-- ============================================================ reports
-- Per-user saved diagnostic reports (private, RLS-protected).
create table if not exists public.reports (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    biological_age integer,
    frailty_indicator text,
    anomalies jsonb default '[]'::jsonb,
    recommendations jsonb default '[]'::jsonb,
    voice_stats jsonb default '{}'::jsonb,
    gait_stats jsonb default '{}'::jsonb,
    voice_source text,
    gait_source text,
    created_at timestamptz default now()
);

create index if not exists reports_user_created_idx
    on public.reports (user_id, created_at desc);

alter table public.reports enable row level security;

-- Users may only read and write their own rows.
drop policy if exists "reports_select_own" on public.reports;
create policy "reports_select_own"
    on public.reports for select
    using (auth.uid() = user_id);

drop policy if exists "reports_insert_own" on public.reports;
create policy "reports_insert_own"
    on public.reports for insert
    with check (auth.uid() = user_id);

drop policy if exists "reports_delete_own" on public.reports;
create policy "reports_delete_own"
    on public.reports for delete
    using (auth.uid() = user_id);


-- ================================================== anonymized_results
-- Opt-in derived biomarker numbers only. No identity, no raw media.
create table if not exists public.anonymized_results (
    id uuid primary key default gen_random_uuid(),
    biological_age integer,
    frailty_indicator text,
    anomaly_count integer,
    jitter_proxy double precision,
    shimmer_proxy double precision,
    spectral_centroid_mean double precision,
    asymmetry_index double precision,
    step_frequency double precision,
    voice_source text,
    gait_source text,
    voice_is_mock boolean default false,
    gait_is_mock boolean default false,
    created_at timestamptz default now()
);

alter table public.anonymized_results enable row level security;

-- Anyone (including anonymous visitors) may contribute a row.
drop policy if exists "anon_insert_anonymized" on public.anonymized_results;
create policy "anon_insert_anonymized"
    on public.anonymized_results for insert
    with check (true);

-- Anyone may read aggregate rows for the community stats panel.
drop policy if exists "anon_select_anonymized" on public.anonymized_results;
create policy "anon_select_anonymized"
    on public.anonymized_results for select
    using (true);
