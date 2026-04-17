#!/usr/bin/env node
/**
 * Hexo → Astro interview post converter
 * Reads 182 Hexo-format .md files and converts them to Astro content collection format.
 */

import fs from 'node:fs';
import path from 'node:path';

// ─── Configuration ───────────────────────────────────────────────────────────

const SOURCE_DIR = path.resolve(
  import.meta.dirname,
  '../../zhengshengning.github.io/source/_posts/面试宝典/AI Infra/AIInfra面经',
);
const OUTPUT_DIR = path.resolve(import.meta.dirname, '../docs/interview');

// Sub-departments that fold into the parent company
const SUB_DEPT_MAP = {
  // 字节跳动
  字节跳动_AML: '字节跳动',
  字节跳动_抖音: '字节跳动',
  字节跳动_抖音电商: '字节跳动',
  字节跳动_豆包: '字节跳动',
  // 阿里巴巴
  阿里巴巴_云: '阿里巴巴',
  阿里巴巴_淘天: '阿里巴巴',
  阿里巴巴_国际: '阿里巴巴',
  阿里巴巴_控股集团: '阿里巴巴',
  // 腾讯
  腾讯_TEG: '腾讯',
  // 美团
  美团_北斗: '美团',
  // 科大讯飞
  科大讯飞_飞星: '科大讯飞',
  // OPPO
  OPPO_云: 'OPPO',
  // 理想汽车
  理想汽车_云: '理想汽车',
};

const TIER_MAP = {
  字节跳动: 'T0',
  阿里巴巴: 'T0',
  腾讯: 'T0',
  百度: 'T0',
  快手: 'T1',
  美团: 'T1',
  蚂蚁: 'T1',
  京东: 'T1',
  华为: 'T1',
  OPPO: 'T1',
  小米: 'T1',
  拼多多: 'T1',
  B站: 'T1',
  MiniMax: 'T2',
  阶跃星辰: 'T2',
  智谱: 'T2',
  商汤: 'T2',
  旷视科技: 'T2',
  英伟达: 'T3',
  壁仞科技: 'T3',
  寒武纪: 'T3',
  摩尔线程: 'T3',
  飞腾: 'T3',
  沐曦: 'T3',
  三星: 'T3',
  太初: 'T3',
  遂原科技: 'T3',
  燧原科技: 'T3',
  北极雄芯: 'T3',
  后摩智能: 'T3',
  原粒半导体: 'T3',
  海光: 'T3',
  蔚来: 'T4',
  小鹏汽车: 'T4',
  理想汽车: 'T4',
  大疆车载: 'T4',
  卓驭: 'T4',
  文远知行: 'T4',
  元戎启行: 'T4',
  小马智行: 'T4',
  辉羲智能: 'T4',
  易控智驾: 'T4',
  综合: '综合',
};

// Tier order base for the `order` field (ensures T0 sorts before T1 etc.)
const TIER_ORDER_BASE = {
  综合: 0,
  T0: 1000,
  T1: 2000,
  T2: 3000,
  T3: 4000,
  T4: 5000,
  T5: 6000,
};

// Tags to filter out (generic, non-technical)
const GENERIC_TAGS = new Set([
  'AIInfra',
  'AI Infra',
  '面经',
  '大厂面经',
  '求职面试',
]);

// Special prefix patterns → company "综合"
const ZONGHE_PREFIXES = [
  'AI_Infra_综合面经题库',
  'AI_Infra_面经',
  'AI_Infra_校招',
  'AI_Infra_一面',
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function parseFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return { frontmatter: {}, body: raw };

  const fm = {};
  const fmRaw = match[1];
  const body = raw.slice(match[0].length);

  // Parse simple YAML-like fields
  for (const line of fmRaw.split('\n')) {
    const kvMatch = line.match(/^(\w+):\s*(.*)$/);
    if (kvMatch) {
      const [, key, val] = kvMatch;
      // Handle inline array: [a, b, c]
      if (val.startsWith('[') && val.endsWith(']')) {
        fm[key] = val
          .slice(1, -1)
          .split(',')
          .map((s) => s.trim().replace(/^['"]|['"]$/g, ''));
      } else {
        fm[key] = val.replace(/^['"]|['"]$/g, '');
      }
    }
  }

  return { frontmatter: fm, body };
}

function extractCompany(filename) {
  const stem = filename.replace(/\.md$/, '');

  // Check "综合" special prefixes
  for (const prefix of ZONGHE_PREFIXES) {
    if (stem.startsWith(prefix)) return '综合';
  }

  // 小厂
  if (stem.startsWith('小厂_')) return '小厂';

  // Extract company: everything before _AI_Infra
  const aiIdx = stem.indexOf('_AI_Infra');
  if (aiIdx === -1) return '综合';

  const rawCompany = stem.slice(0, aiIdx);

  // Fold sub-departments
  if (SUB_DEPT_MAP[rawCompany]) return SUB_DEPT_MAP[rawCompany];

  return rawCompany;
}

function extractInterviewType(title) {
  if (title.includes('实习')) return '实习';
  if (title.includes('校招')) return '校招';
  if (title.includes('社招')) return '社招';
  return '未知';
}

function extractRound(title) {
  // Match patterns like 一面, 二面, 三面, 一二面, 一二三面
  const m = title.match(/([一二三四五六七八九十]+面)/);
  return m ? m[1] : undefined;
}

function filterTags(tags) {
  if (!Array.isArray(tags)) return [];
  return tags.filter((t) => !GENERIC_TAGS.has(t));
}

function cleanBody(body) {
  // Remove <!-- more --> and the --- separator right after it
  return body
    .replace(/<!--\s*more\s*-->\s*\n*---\s*\n*/g, '')
    .replace(/<!--\s*more\s*-->\s*\n*/g, '')
    .trim();
}

function extractDate(dateStr) {
  if (!dateStr) return '2026-04-15';
  // Handle "2026-04-15 12:00:00" → "2026-04-15"
  return dateStr.split(' ')[0];
}

// ─── Main ────────────────────────────────────────────────────────────────────

function main() {
  if (!fs.existsSync(SOURCE_DIR)) {
    console.error(`Source directory not found: ${SOURCE_DIR}`);
    process.exit(1);
  }

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const files = fs.readdirSync(SOURCE_DIR).filter((f) => f.endsWith('.md'));
  console.log(`Found ${files.length} source files`);

  // First pass: collect all entries to assign order within each company
  const entries = [];
  for (const file of files) {
    const raw = fs.readFileSync(path.join(SOURCE_DIR, file), 'utf-8');
    const { frontmatter, body } = parseFrontmatter(raw);

    const company = extractCompany(file);
    const tier = TIER_MAP[company] || 'T5';
    const title = frontmatter.title || file.replace(/\.md$/, '').replace(/_/g, ' ');
    const interviewType = extractInterviewType(title);
    const round = extractRound(title);
    const tags = filterTags(frontmatter.tags);
    const pubDate = extractDate(frontmatter.date);
    const cleanedBody = cleanBody(body);

    const topicStr = tags.length > 0 ? tags.join('、') : 'AI Infra';
    const roundStr = round ? round : '';
    const description = `${company === '综合' ? '' : company + ' '}AI Infra ${interviewType !== '未知' ? interviewType : ''}${roundStr}面试真题，涵盖${topicStr}等方向`;

    const outFilename = file.replace(/_/g, '-');

    entries.push({
      file,
      outFilename,
      company,
      tier,
      title,
      description,
      pubDate,
      interviewType,
      round,
      tags,
      body: cleanedBody,
    });
  }

  // Sort by tier then company for order assignment
  entries.sort((a, b) => {
    const tierA = TIER_ORDER_BASE[a.tier] ?? 6000;
    const tierB = TIER_ORDER_BASE[b.tier] ?? 6000;
    if (tierA !== tierB) return tierA - tierB;
    return a.company.localeCompare(b.company, 'zh-CN');
  });

  // Assign order: tier base + sequential within tier
  const tierCounters = {};
  for (const entry of entries) {
    const base = TIER_ORDER_BASE[entry.tier] ?? 6000;
    tierCounters[entry.tier] = (tierCounters[entry.tier] || 0) + 1;
    entry.order = base + tierCounters[entry.tier];
  }

  // Write output files
  let written = 0;
  for (const entry of entries) {
    const tagsYaml = entry.tags.length > 0
      ? `[${entry.tags.map((t) => `"${t}"`).join(', ')}]`
      : '[]';

    const roundLine = entry.round ? `round: "${entry.round}"\n` : '';

    const output = `---
title: "${entry.title}"
description: "${entry.description}"
pubDate: ${entry.pubDate}
company: "${entry.company}"
tier: "${entry.tier}"
interviewType: "${entry.interviewType}"
${roundLine}order: ${entry.order}
tags: ${tagsYaml}
---

${entry.body}
`;

    fs.writeFileSync(path.join(OUTPUT_DIR, entry.outFilename), output, 'utf-8');
    written++;
  }

  console.log(`Converted ${written} files to ${OUTPUT_DIR}`);

  // Print summary
  const companySet = new Set(entries.map((e) => e.company));
  const tierSet = new Set(entries.map((e) => e.tier));
  console.log(`Companies: ${companySet.size}, Tiers: ${tierSet.size}`);
  for (const tier of ['综合', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5']) {
    const tierEntries = entries.filter((e) => e.tier === tier);
    if (tierEntries.length > 0) {
      const companies = [...new Set(tierEntries.map((e) => e.company))];
      console.log(`  ${tier}: ${tierEntries.length} articles (${companies.join(', ')})`);
    }
  }
}

main();
