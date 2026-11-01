// Wrap every Markdown <table> in <div class="table-wrap"> so wide tables
// scroll inside their own box instead of widening the whole page on mobile.
export default function rehypeTableWrap() {
  const walk = (node) => {
    if (!node.children) return;
    node.children = node.children.map((child) => {
      if (child.type === 'element' && child.tagName === 'table') {
        return {
          type: 'element',
          tagName: 'div',
          properties: { className: ['table-wrap'], tabIndex: 0, role: 'region', ariaLabel: '表格' },
          children: [child],
        };
      }
      walk(child);
      return child;
    });
  };
  return (tree) => walk(tree);
}
