/** True when a turn carries an image part and the loaded model cannot see images. */
export function isImagePart(p: any): boolean {
  return !!p && p.type === "file" && typeof p.mime === "string" && p.mime.startsWith("image/");
}

export function needsVision(parts: any[], status: { vision?: boolean } | undefined): boolean {
  if (!status || status.vision !== false) return false;
  return (parts ?? []).some(isImagePart);
}
