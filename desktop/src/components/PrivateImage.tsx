import { useEffect, useRef, useState, type ImgHTMLAttributes } from "react";
import { get } from "../api";

export default function PrivateImage({ id, ...props }: ImgHTMLAttributes<HTMLImageElement> & { id: string }) {
  const element = useRef<HTMLSpanElement>(null);
  const [requested, setRequested] = useState(false);
  const [image, setImage] = useState<{ id: string; src: string }>();
  useEffect(() => {
    if (!element.current) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setRequested(true);
        observer.disconnect();
      }
    }, { rootMargin: "120px" });
    observer.observe(element.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!requested) return;
    let active = true;
    get<string>(`/api/screenshots/${id}/image`)
      .then((src) => { if (active) setImage({ id, src }); })
      .catch(() => {});
    return () => { active = false; };
  }, [id, requested]);
  return image?.id === id ? <img {...props} src={image.src} /> : <span ref={element} className={props.className} style={{ display: "block", minHeight: 16 }} />;
}
