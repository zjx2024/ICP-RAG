import React from 'react';
import { Button, Typography } from 'antd';

/* ============================================================
   白毛玻璃 UI 组件库（v2 形态层）
   改造前后对比：
   - Button(danger/默认)      → GlassButton：玻璃描边红 / 玻璃悬浮主按钮 / 文字+下划线
   - Card+Statistic           → GlassStatCard：大号玻璃角标（hover 浮起）
   - antd Empty               → GlassEmpty：玻璃占位卡（柔和虚线边框）
   - radio/checkbox 大列表    → GlassOptionCard：点击整卡选中（玻璃高亮边框）
   - 零散操作区               → GlassToolbar：底部固定玻璃工具栏
   ============================================================ */

export type GlassTone = 'primary' | 'danger' | 'text' | 'default';

type GlassButtonProps = Omit<React.ComponentProps<typeof Button>, 'variant'> & {
  tone?: GlassTone;
  variant?: GlassTone;
};

/* 设计决策：页面统一使用 `variant` 语义区分玻璃按钮形态。
   这里在解构时吞掉 antd Button 原生 variant 属性（5.17+ 新增），
   否则 `variant="primary"` 会当作 antd 内置 variant 透传导致玻璃样式失效，
   且 primary/danger 不在 antd variant 合法取值内（TS 报错）。
   类型上用 Omit 去掉 antd 原生 variant，避免与我们的 variant 并集取交集只剩 'text'。 */
export function GlassButton({
  tone,
  variant,
  className,
  children,
  ...rest
}: GlassButtonProps) {
  const t = tone || variant || 'default';
  const cls = {
    primary: 'glass-btn-primary',
    danger: 'glass-btn-danger',
    text: 'glass-btn-text',
    default: '',
  }[t];
  return (
    <Button
      type={t === 'primary' ? 'primary' : 'default'}
      className={[cls, className].filter(Boolean).join(' ')}
      {...rest}
    >
      {children}
    </Button>
  );
}

export function GlassStatCard({
  label,
  value,
  suffix,
}: {
  label: string;
  value: React.ReactNode;
  suffix?: string;
}) {
  return (
    <div className="glass-stat">
      <div className="value">
        {value}
        {suffix ? <span style={{ fontSize: 16, fontWeight: 600, color: '#595959', marginLeft: 4 }}>{suffix}</span> : null}
      </div>
      <div className="label">{label}</div>
    </div>
  );
}

export function GlassToolbar({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div className="glass-toolbar" style={style}>
      {children}
    </div>
  );
}

export function GlassEmpty({
  description,
  children,
}: {
  description?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <div className="glass-empty">
      <Typography.Text type="secondary">{description || '暂无数据'}</Typography.Text>
      {children}
    </div>
  );
}

export function GlassOptionCard({
  selected,
  disabled,
  onClick,
  children,
  style,
}: {
  selected?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={['glass-option-card', selected ? 'selected' : '', disabled ? 'disabled' : '']
        .filter(Boolean)
        .join(' ')}
      onClick={disabled ? undefined : onClick}
      role="button"
      style={style}
    >
      {children}
    </div>
  );
}
