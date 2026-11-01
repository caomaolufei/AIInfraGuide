// Wrap everything between a "参考资料" <h2> and the next <h2> in
// <section class="references"> so the reference list can use a smaller,
// muted style without touching the rest of the article.
const textOf = (node) =>
  node.type === 'text' ? node.value : (node.children ?? []).map(textOf).join('');

const isH2 = (node) => node.type === 'element' && node.tagName === 'h2';

export default function rehypeReferences() {
  return (tree) => {
    const out = [];
    let section = null;
    for (const child of tree.children) {
      if (isH2(child)) {
        section = null;
        out.push(child);
        if (textOf(child).includes('参考资料')) {
          section = {
            type: 'element',
            tagName: 'section',
            properties: { className: ['references'] },
            children: [],
          };
          out.push(section);
        }
        continue;
      }
      if (section) section.children.push(child);
      else out.push(child);
    }
    tree.children = out;
  };
}
