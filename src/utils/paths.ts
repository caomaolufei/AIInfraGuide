export function getPath(path: string): string {
  const base = import.meta.env.BASE_URL;
  return base === '/' ? path : `${base.replace(/\/$/, '')}${path}`;
}
