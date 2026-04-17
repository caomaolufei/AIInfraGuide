import type { CollectionEntry } from 'astro:content';

type Interview = CollectionEntry<'interview'>;

export interface CompanyGroup {
  company: string;
  articles: Interview[];
}

export interface TierGroup {
  tier: string;
  label: string;
  color: string;
  companies: CompanyGroup[];
}

export const TIER_META: Record<string, { label: string; color: string }> = {
  T0: { label: 'T0 大厂', color: '#EF4444' },
  T1: { label: 'T1 大厂/独角兽', color: '#F59E0B' },
  T2: { label: 'T2 AI 独角兽', color: '#8B5CF6' },
  T3: { label: 'T3 芯片/硬件', color: '#06B6D4' },
  T4: { label: 'T4 车企/自驾', color: '#10B981' },
  T5: { label: 'T5 其他', color: '#6B7280' },
  综合: { label: '综合面经', color: '#3B82F6' },
};

const TIER_ORDER = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', '综合'];

/**
 * Group interview entries by tier → company.
 * Returns an array of TierGroup sorted by tier order,
 * each containing CompanyGroups sorted by article count descending.
 */
export function groupByCompany(items: Interview[]): TierGroup[] {
  const sorted = [...items].sort((a, b) => a.data.order - b.data.order);

  // tier → company → articles
  const tierMap = new Map<string, Map<string, Interview[]>>();

  for (const item of sorted) {
    const tier = item.data.tier;
    const company = item.data.company;
    if (!tierMap.has(tier)) tierMap.set(tier, new Map());
    const companyMap = tierMap.get(tier)!;
    if (!companyMap.has(company)) companyMap.set(company, []);
    companyMap.get(company)!.push(item);
  }

  const result: TierGroup[] = [];

  for (const tier of TIER_ORDER) {
    const companyMap = tierMap.get(tier);
    if (!companyMap) continue;

    const meta = TIER_META[tier] ?? { label: tier, color: '#6B7280' };
    const companies: CompanyGroup[] = [...companyMap.entries()]
      .map(([company, articles]) => ({ company, articles }))
      .sort((a, b) => b.articles.length - a.articles.length);

    result.push({ tier, ...meta, companies });
  }

  return result;
}

/**
 * Flatten all interviews into reading order: sorted by tier order, then by `order` field.
 */
export function flattenCompanyOrder(items: Interview[]): Interview[] {
  return [...items].sort((a, b) => a.data.order - b.data.order);
}
