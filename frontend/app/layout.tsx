import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
    title: 'HawkEye AI | Intelligent Surveillance',
    description: 'Next-generation AI-powered surveillance dashboard',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className={`${inter.className} bg-dark-900 text-white min-h-screen`}>
                {children}
            </body>
        </html>
    );
}
