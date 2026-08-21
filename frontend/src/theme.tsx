import React from 'react';
import { ConfigProvider, theme as antdTheme } from 'antd';
import zhCN from 'antd/locale/zh_CN';

// ===== 黑白玻璃主题（白多黑少）=====
// 品牌紫 #667eea/#764ba2 全部替换为炭黑 #1a1a1a，文字统一 antd 灰色阶，杜绝花哨

// 1. 语义 token：颜色、圆角、字体
export const glassToken = {
  colorPrimary: '#1a1a1a',
  colorInfo: '#1a1a1a',
  colorBgLayout: 'transparent', // 布局背景交给 body 渐变
  colorBgContainer: 'rgba(255,255,255,0.60)',
  colorBgElevated: 'rgba(255,255,255,0.80)',
  colorBorder: 'rgba(255,255,255,0.65)',
  colorBorderSecondary: 'rgba(255,255,255,0.65)',
  borderRadius: 12,
  colorText: '#262626',
  colorTextSecondary: '#595959',
  colorTextTertiary: '#8c8c8c',
  fontFamily: "-apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif",
  fontSize: 14,
  lineWidth: 1,
  colorLink: '#1a1a1a',
};

// 2. 覆盖关键组件为玻璃质感
export const glassComponents = {
  Layout: { bodyBg: 'transparent', siderBg: 'rgba(255,255,255,0.55)', headerBg: 'rgba(255,255,255,0.55)' },
  Card: {
    colorBgContainer: 'rgba(255,255,255,0.60)',
    boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
    borderRadiusLG: 16,
  },
  Modal: { contentBg: 'rgba(255,255,255,0.72)', borderRadiusLG: 16, headerBg: 'transparent' },
  Drawer: { colorBgElevated: 'rgba(255,255,255,0.72)' },
  Table: {
    headerBg: 'rgba(255,255,255,0.50)',
    rowHoverBg: 'rgba(0,0,0,0.04)',
    borderColor: 'rgba(255,255,255,0.65)',
    colorBgContainer: 'transparent',
  },
  Menu: {
    itemBg: 'transparent',
    itemSelectedBg: 'rgba(0,0,0,0.08)',
    itemSelectedColor: '#1a1a1a',
    itemColor: '#262626',
    itemHoverBg: 'rgba(0,0,0,0.05)',
  },
  Input: { colorBgContainer: 'rgba(255,255,255,0.55)', activeBorderColor: '#1a1a1a', activeShadow: '0 0 0 2px rgba(0,0,0,0.10)' },
  InputNumber: { colorBgContainer: 'rgba(255,255,255,0.55)', activeBorderColor: '#1a1a1a', activeShadow: '0 0 0 2px rgba(0,0,0,0.10)' },
  Select: { colorBgContainer: 'rgba(255,255,255,0.55)', optionSelectedBg: 'rgba(0,0,0,0.06)' },
  Button: { primaryShadow: '0 6px 16px rgba(0,0,0,0.20)' },
  Tabs: { itemSelectedColor: '#1a1a1a', inkBarColor: '#1a1a1a' },
  Tag: { defaultBg: 'rgba(255,255,255,0.70)', defaultColor: '#262626' },
  Progress: { remainingColor: 'rgba(255,255,255,0.40)' },
  Segmented: {
    itemSelectedBg: 'rgba(255,255,255,0.85)',
    itemSelectedColor: '#1a1a1a',
    itemColor: '#595959',
    itemHoverColor: '#262626',
    trackBg: 'rgba(255,255,255,0.45)',
    itemHoverBg: 'rgba(255,255,255,0.6)',
  },
  Radio: { buttonColor: '#595959', buttonSolidCheckedBg: '#1a1a1a', buttonSolidCheckedActiveBg: '#000000' },
  Checkbox: { colorPrimary: '#1a1a1a' },
  Switch: { colorPrimary: '#1a1a1a' },
  Message: {
    contentBg: 'rgba(255,255,255,0.75)',
    borderRadiusLG: 999,
    contentPadding: '12px 20px',
    boxShadow: '0 10px 36px rgba(0,0,0,0.14)',
    colorText: '#262626',
    colorIcon: '#595959',
  },
  Empty: { colorTextDisabled: '#8c8c8c', fontSize: 13 },
  Spin: { colorPrimary: '#1a1a1a' },
  Popconfirm: { colorBgElevated: 'rgba(255,255,255,0.78)', borderRadiusLG: 14 },
  Dropdown: { colorBgElevated: 'rgba(255,255,255,0.78)', borderRadiusLG: 14 },
  Tooltip: { colorBgSpotlight: 'rgba(26,26,26,0.92)', colorTextLightSolid: '#fff', borderRadius: 8 },
  Slider: { trackBg: '#1a1a1a', handleColor: '#1a1a1a', railBg: 'rgba(255,255,255,0.6)', handleActiveColor: '#000000' },
  Timeline: { tailColor: 'rgba(0,0,0,0.12)', dotBg: 'rgba(255,255,255,0.8)' },
};

export default function AppTheme({ children }: { children: React.ReactNode }) {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: antdTheme.defaultAlgorithm,
        token: glassToken,
        components: glassComponents,
      }}
    >
      {children}
    </ConfigProvider>
  );
}
